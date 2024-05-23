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

import scala.collection.mutable

import ai.rapids.cudf.{HostColumnVector, NvtxColor}
import com.nvidia.spark.rapids.{CoalesceSizeGoal, CudfRowTransitions, GeneratedInternalRowToCudfRowIterator, GpuColumnVector, GpuMetric, GpuRowToColumnConverter, GpuSemaphore, NvtxWithMetrics, RowToColumnarIterator}
import com.nvidia.spark.rapids.Arm._
import com.nvidia.spark.rapids.RapidsPluginImplicits.AutoCloseableProducingArray
import com.nvidia.spark.rapids.velox.VeloxBatchConverter
import io.glutenproject.execution.VeloxColumnarToRowExec

import org.apache.spark.TaskContext
import org.apache.spark.internal.Logging
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.Attribute
import org.apache.spark.sql.types.{StructField, StructType}
import org.apache.spark.sql.vectorized.{ColumnarBatch, ColumnVector}


object VeloxColumnarBatchConverter extends Logging {

  private class CoalesceNativeConverter(veloxIter: Iterator[ColumnarBatch],
                                        targetBatchSizeInBytes: Int,
                                        schema: StructType,
                                        metrics: Map[String, GpuMetric])
    extends Iterator[Array[HostColumnVector]] {

    private var converterImpl: Option[VeloxBatchConverter] = None

    private val deck = mutable.Queue[ColumnarBatch]()

    private val c2cMetrics = Map(
      "VeloxC2CConvertTime" -> GpuMetric.unwrap(metrics("VeloxC2CConvertTime")),
      "OutputSizeInBytes" -> GpuMetric.unwrap(metrics("OutputSizeInBytes")))

    override def hasNext(): Boolean = {
      converterImpl.exists(_.hasStackedBatches) || !sourceExhausted
    }

    override def next(): Array[HostColumnVector] = {
      // require(hasNext(), "try to call next of an exhausted CoalesceNativeConverter")

      while (true) {
        converterImpl.foreach { impl =>
          if (impl.readyToFlush || sourceExhausted) {
            withResource(new NvtxWithMetrics("VeloxC2C", NvtxColor.YELLOW,
              metrics("VeloxC2CTime"))) { _ =>
              metrics("C2COutputBatches") += 1
              return impl.flushAndConvert()
            }
          }
        }

        val srcBatch = if (deck.isEmpty) {
          metrics("VeloxOutputBatches") += 1
          veloxIter.next()
        } else {
          deck.dequeue()
        }

        withResource(new NvtxWithMetrics("VeloxC2C", NvtxColor.YELLOW,
          metrics("VeloxC2CTime"))) { _ =>

          if (converterImpl.isEmpty) {
            converterImpl = Some(
              VeloxBatchConverter(srcBatch, targetBatchSizeInBytes, schema, c2cMetrics))
          } else {
            if (!converterImpl.get.tryAppendBatch(srcBatch)) {
              deck.enqueue(srcBatch)
            }
          }
        }
      }

      throw new RuntimeException("should NOT reach this line")
    }

    private def sourceExhausted: Boolean = deck.isEmpty && !veloxIter.hasNext
  }

  def nativeConvert(iter: Iterator[ColumnarBatch],
      outputAttr: Seq[Attribute],
      coalesceGoal: CoalesceSizeGoal,
      metrics: Map[String, GpuMetric]): Iterator[ColumnarBatch] = {
    val schema = StructType(outputAttr.map { ar =>
      StructField(ar.name, ar.dataType, ar.nullable)
    })
    val dataTypes = outputAttr.map(_.dataType).toArray

    require(coalesceGoal.targetSizeBytes <= Int.MaxValue,
      s"targetSizeBytes should be smaller than 2GB, but got ${coalesceGoal.targetSizeBytes}")
    val hostIter = new CoalesceNativeConverter(
      iter, coalesceGoal.targetSizeBytes.toInt, schema, metrics)

    hostIter.map { hostVectors =>
      withResource(new NvtxWithMetrics("gpuAcquireC2C", NvtxColor.WHITE,
        metrics("gpuAcquireTime"))) { _ =>
        Option(TaskContext.get()).foreach(GpuSemaphore.acquireIfNecessary)
      }

      withResource(new NvtxWithMetrics("HostToDeviceC2C", NvtxColor.BLUE,
        metrics("H2DTime"))) { _ =>
        val deviceVectors: Array[ColumnVector] =
          hostVectors.zip(dataTypes).safeMap { case (hcv, dt) =>
            withResource(hcv) { _ =>
              GpuColumnVector.from(hcv.copyToDevice(), dt)
            }
          }
        new ColumnarBatch(deviceVectors, hostVectors.head.getRowCount.toInt)
      }
    }
  }

  def roundTripConvert(iter: Iterator[ColumnarBatch],
                       outputAttr: Seq[Attribute],
                       coalesceGoal: CoalesceSizeGoal,
                       metrics: Map[String, GpuMetric]): Iterator[ColumnarBatch] = {
    val rowIter: Iterator[InternalRow] = VeloxColumnarToRowExec.toRowIterator(
      iter,
      outputAttr,
      GpuMetric.unwrap(metrics("C2ROutputRows")),
      GpuMetric.unwrap(metrics("C2ROutputBatches")),
      GpuMetric.unwrap(metrics("VeloxC2RTime"))
    )
    val useCudfRowTransition: Boolean = {
      outputAttr.nonEmpty && outputAttr.length < 100000000 &&
        CudfRowTransitions.areAllSupported(outputAttr)
    }

    if (useCudfRowTransition) {
      GeneratedInternalRowToCudfRowIterator(
        rowIter,
        outputAttr.toArray, coalesceGoal,
        metrics("R2CStreamTime"), metrics("R2CTime"),
        metrics("R2CInputRows"), metrics("R2COutputRows"), metrics("R2COutputBatches"))
    } else {
      val fullOutputSchema = StructType(outputAttr.map { ar =>
        StructField(ar.name, ar.dataType, ar.nullable)
      })
      val converters = new GpuRowToColumnConverter(fullOutputSchema)
      new RowToColumnarIterator(
        rowIter,
        fullOutputSchema, coalesceGoal, converters,
        metrics("R2CInputRows"), metrics("R2COutputRows"), metrics("R2COutputBatches"),
        metrics("R2CStreamTime"), metrics("R2CTime"), metrics("gpuAcquireTime"))
    }
  }

}
