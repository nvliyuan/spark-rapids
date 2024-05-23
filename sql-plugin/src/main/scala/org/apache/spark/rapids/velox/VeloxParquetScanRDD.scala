/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

package org.apache.spark.rapids.velox

import ai.rapids.cudf.{NvtxColor, NvtxRange}
import com.nvidia.spark.rapids.{CoalesceSizeGoal, GpuMetric}
import com.nvidia.spark.rapids.Arm.withResource

import org.apache.spark.{InterruptibleIterator, Partition, TaskContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.Attribute
import org.apache.spark.sql.execution.metric.SQLMetric
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.vectorized.ColumnarBatch

class VeloxParquetScanRDD(scanRDD: RDD[ColumnarBatch],
                          outputAttr: Seq[Attribute],
                          outputSchema: StructType,
                          coalesceGoal: CoalesceSizeGoal,
                          useNativeConverter: Boolean,
                          @transient metrics: Map[String, GpuMetric])
  extends RDD[InternalRow](scanRDD.sparkContext, Nil) {

  private val veloxScanTime = GpuMetric.unwrap(metrics("veloxScanTime"))

  private val convertMetrics = if (useNativeConverter) {
    Map(
      "gpuAcquireTime" -> metrics("gpuAcquireTime"),
      "VeloxC2CTime" -> metrics("VeloxC2CTime"),
      "VeloxC2CConvertTime" -> metrics("VeloxC2CConvertTime"),
      "OutputSizeInBytes" -> metrics("OutputSizeInBytes"),
      "CoalesceConcatTime" -> metrics("CoalesceConcatTime"),
      "CoalesceOpTime" -> metrics("CoalesceOpTime"),
      "H2DTime" -> metrics("H2DTime"),
      "C2COutputBatches" -> metrics("C2COutputBatches"),
      "VeloxOutputBatches" -> metrics("VeloxOutputBatches"),
    )
  } else {
    Map(
      "C2ROutputRows" -> metrics("C2ROutputRows"),
      "C2ROutputBatches" -> metrics("C2ROutputBatches"),
      "VeloxC2RTime" -> metrics("VeloxC2RTime"),
      "gpuAcquireTime" -> metrics("gpuAcquireTime"),
      "R2CStreamTime" -> metrics("R2CStreamTime"),
      "R2CTime" -> metrics("R2CTime"),
      "R2CInputRows" -> metrics("R2CInputRows"),
      "R2COutputRows" -> metrics("R2COutputRows"),
      "R2COutputBatches" -> metrics("R2COutputBatches"),
    )
  }

  override protected def getPartitions: Array[Partition] = scanRDD.partitions

  override def compute(split: Partition, context: TaskContext): Iterator[InternalRow] = {
    val veloxCbIter = new VeloxScanMetricsIter(
      scanRDD.compute(split, context),
      veloxScanTime
    )
    val deviceIter = if (useNativeConverter) {
      VeloxColumnarBatchConverter.nativeConvert(
        veloxCbIter, outputAttr, coalesceGoal, convertMetrics)
    } else {
      VeloxColumnarBatchConverter.roundTripConvert(
        veloxCbIter, outputAttr, coalesceGoal, convertMetrics)
    }

    // TODO: SPARK-25083 remove the type erasure hack in data source scan
    new InterruptibleIterator(context, deviceIter.asInstanceOf[Iterator[InternalRow]])
  }
}

private class VeloxScanMetricsIter(iter: Iterator[ColumnarBatch],
                                   scanTime: SQLMetric
                                  ) extends Iterator[ColumnarBatch] {
  override def hasNext: Boolean = {
    val start = System.nanoTime()
    try {
      withResource(new NvtxRange("velox scan hasNext", NvtxColor.WHITE)) { _ =>
        iter.hasNext
      }
    } finally {
      scanTime += System.nanoTime() - start
    }
  }

  override def next(): ColumnarBatch = {
    val start = System.nanoTime()
    try {
      withResource(new NvtxRange("velox scan next", NvtxColor.BLUE)) { _ =>
        iter.next()
      }
    } finally {
      scanTime += System.nanoTime() - start
    }
  }
}
