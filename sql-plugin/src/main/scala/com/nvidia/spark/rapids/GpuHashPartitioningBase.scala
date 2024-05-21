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

import ai.rapids.cudf.{DType, NvtxColor, NvtxRange, Table}
import com.nvidia.spark.rapids.Arm.withResource
import com.nvidia.spark.rapids.jni.TableOperation
import com.nvidia.spark.rapids.shims.ShimExpression

import org.apache.spark.sql.catalyst.expressions.Expression
import org.apache.spark.sql.rapids.GpuMurmur3Hash
import org.apache.spark.sql.types.{DataType, IntegerType}
import org.apache.spark.sql.vectorized.ColumnarBatch

abstract class GpuHashPartitioningBase(expressions: Seq[Expression],
                                       numPartitions: Int,
                                       enablePaddingPartition: Boolean = false)
  extends GpuExpression with ShimExpression with GpuPartitioning with Serializable {

  override def children: Seq[Expression] = expressions
  override def nullable: Boolean = false
  override def dataType: DataType = IntegerType
  override def usePaddingPartition: Boolean = enablePaddingPartition

  def partitionInternalAndClose(batch: ColumnarBatch): (Array[Int], Array[GpuColumnVector]) = {
    val types = GpuColumnVector.extractTypes(batch)
    val (pt, parts) = GpuHashPartitioningBase.hashPartitionAndClose(batch, expressions,
      numPartitions, usePaddingPartition && !usesGPUShuffle, "Calculate part")
    withResource(pt) { partedTable =>
      val columns = (0 until partedTable.getNumberOfColumns).zip(types).map {
        case (index, sparkType) =>
          GpuColumnVector.from(partedTable.getColumn(index).incRefCount(), sparkType)
      }
      (parts, columns.toArray)
    }
  }

  override def columnarEvalAny(batch: ColumnarBatch): Any = {
    //  We are doing this here because the cudf partition command is at this level
    withResource(new NvtxRange("Hash partition", NvtxColor.PURPLE)) { _ =>
      val numRows = batch.numRows
      val (partitionIndexes, partitionColumns) = {
        withResource(new NvtxRange("partition", NvtxColor.BLUE)) { _ =>
          partitionInternalAndClose(batch)
        }
      }
      sliceInternalGpuOrCpuAndClose(numRows, partitionIndexes, partitionColumns)
    }
  }
}

object GpuHashPartitioningBase {

  val DEFAULT_HASH_SEED: Int = 42

  def hashPartitionAndClose(batch: ColumnarBatch, keys: Seq[Expression], numPartitions: Int,
      paddingPartition: Boolean,
      nvtxName: String,
      seed: Int = DEFAULT_HASH_SEED): (Table, Array[Int]) = {
    val sb = SpillableColumnarBatch(batch, SpillPriorities.ACTIVE_ON_DECK_PRIORITY)
    RmmRapidsRetryIterator.withRetryNoSplit(sb) { sb =>
      withResource(sb.getColumnarBatch()) { cb =>
        val parts = withResource(new NvtxRange(nvtxName, NvtxColor.CYAN)) { _ =>
          withResource(GpuMurmur3Hash.compute(cb, keys, seed)) { hash =>
            withResource(GpuScalar.from(numPartitions, IntegerType)) { partsLit =>
              hash.pmod(partsLit, DType.INT32)
            }
          }
        }
        withResource(parts) { parts =>
          withResource(GpuColumnVector.from(cb)) { table =>
            if (paddingPartition) {
              val pt = TableOperation.paddingPartition(table, parts, numPartitions)
              (pt.getTable, pt.getPartitions)
            } else {
              val pt = table.partition(parts, numPartitions)
              (pt.getTable, pt.getPartitions)
            }
          }
        }
      }
    }
  }

}
