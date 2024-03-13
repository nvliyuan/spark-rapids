/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.nvidia.spark.rapids

import scala.collection.mutable.ArrayBuffer

import ai.rapids.cudf.{NvtxColor, NvtxRange}
import com.nvidia.spark.rapids.Arm.{closeOnExcept, withResource}
import com.nvidia.spark.rapids.RapidsPluginImplicits._
import com.nvidia.spark.rapids.shims.ShimUnaryExecNode

import org.apache.spark.TaskContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.Attribute
import org.apache.spark.sql.catalyst.plans.physical.Partitioning
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.types.DataType
import org.apache.spark.sql.vectorized.ColumnarBatch

/**
 * Coalesces serialized tables on the host up to the target batch size before transferring
 * the coalesced result to the GPU. This reduces the overhead of copying data to the GPU
 * and also helps avoid holding onto the GPU semaphore while shuffle I/O is being performed.
 * @note This should ALWAYS appear in the plan after a GPU shuffle when RAPIDS shuffle is
 *       not being used.
 */
case class GpuShuffleCoalesceExec(child: SparkPlan, targetBatchByteSize: Long)
    extends ShimUnaryExecNode with GpuExec {

  import GpuMetric._

  override lazy val additionalMetrics: Map[String, GpuMetric] = Map(
    OP_TIME -> createNanoTimingMetric(MODERATE_LEVEL, DESCRIPTION_OP_TIME),
    NUM_INPUT_ROWS -> createMetric(DEBUG_LEVEL, DESCRIPTION_NUM_INPUT_ROWS),
    NUM_INPUT_BATCHES -> createMetric(DEBUG_LEVEL, DESCRIPTION_NUM_INPUT_BATCHES),
    CONCAT_TIME -> createNanoTimingMetric(DEBUG_LEVEL, DESCRIPTION_CONCAT_TIME)
  )

  override protected val outputBatchesLevel = MODERATE_LEVEL

  override def output: Seq[Attribute] = child.output

  override def outputPartitioning: Partitioning = child.outputPartitioning

  override protected def doExecute(): RDD[InternalRow] = {
    throw new IllegalStateException("ROW BASED PROCESSING IS NOT SUPPORTED")
  }

  override def internalDoExecuteColumnar(): RDD[ColumnarBatch] = {
    val metricsMap = allMetrics
    val targetSize = targetBatchByteSize
    val dataTypes = GpuColumnVector.extractTypes(schema)

    child.executeColumnar().mapPartitions { iter =>
      new GpuShuffleCoalesceIterator(iter, dataTypes, targetSize, metricsMap)
    }
  }
}

/**
 * Iterator that coalesces columnar batches that are expected to only contain
 * [[SerializedTableColumn]]. The serialized tables within are collected up
 * to the target batch size and then concatenated before the data is transferred to the GPU.
 */
class GpuShuffleCoalesceIterator(
    iter: Iterator[ColumnarBatch],
    dataTypes: Array[DataType],
    targetBatchByteSize: Long,
    metricsMap: Map[String, GpuMetric]) extends GpuColumnarBatchIterator(true) {
  private[this] val serializedTables = new ArrayBuffer[SerializedTableColumn]
  private[this] var numTablesInBatch: Int = 0
  private[this] var numRowsInBatch: Int = 0
  private[this] var batchByteSize: Long = 0L
  private[this] val opTime = metricsMap(GpuMetric.OP_TIME)
  private[this] val outputBatches = metricsMap(GpuMetric.NUM_OUTPUT_BATCHES)
  private[this] val outputRows = metricsMap(GpuMetric.NUM_OUTPUT_ROWS)

  override def doClose(): Unit = {
    serializedTables.safeClose()
  }

  override def hasNext: Boolean = {
    if (numTablesInBatch == serializedTables.size) {
      var batchCanGrow = batchByteSize < targetBatchByteSize
      while (batchCanGrow && iter.hasNext) {
        closeOnExcept(iter.next()) { batch =>
          // don't bother tracking empty tables
          if (batch.numRows > 0) {
            val nextTable = batch.column(0).asInstanceOf[SerializedTableColumn]
            serializedTables.append(nextTable)
            val nextTableSize = nextTable.hostBuffer.getLength
            batchCanGrow = batchByteSize + nextTableSize <= targetBatchByteSize &&
              numRowsInBatch.toLong + nextTable.numRows <= Integer.MAX_VALUE
            // always add the first table to the batch even if its beyond the target limits
            if (batchCanGrow || numTablesInBatch == 0) {
              numTablesInBatch += 1
              numRowsInBatch += nextTable.numRows
              batchByteSize += nextTableSize
            }
          } else {
            batch.close()
          }
        }
      }
    }
    numTablesInBatch > 0
  }

  override def next(): ColumnarBatch = {
    if (!hasNext) {
      throw new NoSuchElementException("No more columnar batches")
    }
    withResource(new NvtxRange("Concat Batch", NvtxColor.YELLOW)) { _ =>
      val batchTables = serializedTables.take(numTablesInBatch)
      if (batchTables.head.numColumns == 0) {
        val batch = new ColumnarBatch(Array.empty, numRowsInBatch)
        // We acquire the GPU even on an empty batch, because the downstream tasks expect this
        // iterator to acquire the semaphore and may generate GPU data from batches that are empty.
        GpuSemaphore.acquireIfNecessary(TaskContext.get())
        batch
      } else {
        val startTime = System.nanoTime()
        val sortedTables= batchTables.sortBy(_.hostBuffer.getAddress)
        val headerAddrs = sortedTables.map(_.header.getAddress).toArray
        val dataRanges = computeDataRanges(sortedTables)
        opTime += System.nanoTime() - startTime
        GpuSemaphore.acquireIfNecessary(TaskContext.get())
        opTime.ns {
          val table = ConcatUtil.concatSerializedTables(headerAddrs, dataRanges)
          outputBatches += 1
          outputRows += numRowsInBatch
          numTablesInBatch = 0
          numRowsInBatch = 0
          batchByteSize = 0
          serializedTables.remove(0, numTablesInBatch)
          batchTables.safeClose()
          GpuColumnVector.from(table, dataTypes)
        }
      }
    }
  }

  private def computeDataRanges(tables: ArrayBuffer[SerializedTableColumn]): Array[Long] = {
    val ranges = new ArrayBuffer[Long]
    tables.foreach { table =>
      val tableAddress = table.hostBuffer.getAddress
      val tableSize = table.hostBuffer.getLength
      if (ranges.nonEmpty && ranges.last == tableAddress) {
        ranges.update(ranges.size - 1, tableAddress + tableSize)
        ranges.append(table.hostBuffer.getAddress)
        ranges.append(table.hostBuffer.getLength)
      } else {
        ranges.append(tableAddress)
        ranges.append(tableAddress + tableSize)
      }
    }
    ranges.toArray
  }
}
