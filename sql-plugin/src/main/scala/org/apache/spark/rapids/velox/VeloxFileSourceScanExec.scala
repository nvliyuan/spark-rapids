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

import java.util.concurrent.TimeUnit.NANOSECONDS

import com.nvidia.spark.rapids.{GpuExec, GpuMetric, RapidsConf, TargetSize}
import io.glutenproject.execution.{FileSourceScanExecTransformer, WholeStageTransformer}

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.{InternalRow, TableIdentifier}
import org.apache.spark.sql.catalyst.expressions.{Attribute, DynamicPruningExpression, Expression, Literal}
import org.apache.spark.sql.catalyst.plans.QueryPlan
import org.apache.spark.sql.execution.datasources.HadoopFsRelation
import org.apache.spark.sql.rapids.GpuDataSourceScanExec
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.vectorized.ColumnarBatch
import org.apache.spark.util.collection.BitSet

case class VeloxFileSourceScanExec(
                                    @transient relation: HadoopFsRelation,
                                    originalOutput: Seq[Attribute],
                                    requiredSchema: StructType,
                                    partitionFilters: Seq[Expression],
                                    optionalBucketSet: Option[BitSet],
                                    optionalNumCoalescedBuckets: Option[Int],
                                    dataFilters: Seq[Expression],
                                    tableIdentifier: Option[TableIdentifier],
                                    disableBucketedScan: Boolean = false,
                                    queryUsesInputFile: Boolean = false,
                                    alluxioPathsMap: Option[Map[String, String]])(@transient val rapidsConf: RapidsConf)
  extends GpuDataSourceScanExec with GpuExec {
  import GpuMetric._

  private val glutenScan: FileSourceScanExecTransformer = {
    new FileSourceScanExecTransformer(
      relation,
      originalOutput,
      requiredSchema,
      partitionFilters,
      optionalBucketSet,
      optionalNumCoalescedBuckets,
      dataFilters,
      tableIdentifier,
      disableBucketedScan)
  }

  private val coalesceSizeGoal = rapidsConf.gpuTargetBatchSizeBytes

  override def output: Seq[Attribute] = glutenScan.output

  override lazy val metadata: Map[String, String] = {
    def seqToString(seq: Seq[Any]) = seq.mkString("[", ", ", "]")

    val location = relation.location
    val locationDesc =
      location.getClass.getSimpleName +
        GpuDataSourceScanExec.buildLocationMetadata(location.rootPaths, maxMetadataValueLength)
    Map(
      "Format" -> relation.fileFormat.toString,
      "ReadSchema" -> requiredSchema.catalogString,
      "Batched" -> supportsColumnar.toString,
      "PartitionFilters" -> seqToString(partitionFilters),
      "PushedFilters" -> seqToString(glutenScan.filterExprs()),
      "DataFilters" -> seqToString(dataFilters),
      "Location" -> locationDesc)
  }

  lazy val inputRDD: RDD[InternalRow] = {
    // invoke a whole stage transformer
    val glutenPipeline = WholeStageTransformer(glutenScan, materializeInput = false)(1)

    val glutenScanRDD = glutenPipeline.doExecuteColumnar()

    new VeloxParquetScanRDD(glutenScanRDD,
      output,
      requiredSchema,
      TargetSize(coalesceSizeGoal),
      rapidsConf.enableNativeVeloxConverter,
      allMetrics)
  }

  override def inputRDDs(): Seq[RDD[InternalRow]] = {
    inputRDD :: Nil
  }

  override lazy val allMetrics = Map(
    NUM_OUTPUT_ROWS -> createMetric(ESSENTIAL_LEVEL, DESCRIPTION_NUM_OUTPUT_ROWS),
    NUM_OUTPUT_BATCHES -> createMetric(MODERATE_LEVEL, DESCRIPTION_NUM_OUTPUT_BATCHES),
    "numFiles" -> createMetric(ESSENTIAL_LEVEL, "number of files read"),
    "metadataTime" -> createTimingMetric(ESSENTIAL_LEVEL, "metadata time"),
    "filesSize" -> createSizeMetric(ESSENTIAL_LEVEL, "size of files read"),
    "C2ROutputRows" -> createMetric(MODERATE_LEVEL, "C2ROutputRows"),
    "C2ROutputBatches" -> createMetric(MODERATE_LEVEL, "C2ROutputBatches"),
    "C2COutputBatches" -> createMetric(MODERATE_LEVEL, "C2COutputBatches"),
    "VeloxOutputBatches" -> createMetric(MODERATE_LEVEL, "VeloxOutputBatches"),
    "OutputSizeInBytes" -> createMetric(MODERATE_LEVEL, "OutputSizeInBytes"),
    "R2CInputRows" -> createMetric(MODERATE_LEVEL, "R2CInputRows"),
    "R2COutputRows" -> createMetric(MODERATE_LEVEL, "R2COutputRows"),
    "R2COutputBatches" -> createMetric(MODERATE_LEVEL, "R2COutputBatches"),
    "VeloxC2RTime" -> createTimingMetric(MODERATE_LEVEL, "VeloxC2RTime"),
    "VeloxC2CTime" -> createNanoTimingMetric(MODERATE_LEVEL, "VeloxC2CTime"),
    "VeloxC2CConvertTime" -> createNanoTimingMetric(MODERATE_LEVEL, "VeloxC2CConvertTime"),
    "veloxScanTime" -> createNanoTimingMetric(MODERATE_LEVEL, "veloxScanTime"),
    "R2CStreamTime" -> createNanoTimingMetric(MODERATE_LEVEL, "R2CStreamTime"),
    "gpuAcquireTime" -> createNanoTimingMetric(MODERATE_LEVEL, "gpuAcquireTime"),
    "H2DTime" -> createNanoTimingMetric(MODERATE_LEVEL, "H2DTime"),
    "CoalesceConcatTime" -> createNanoTimingMetric(MODERATE_LEVEL, "CoalesceConcatTime"),
    "CoalesceOpTime" -> createNanoTimingMetric(MODERATE_LEVEL, "CoalesceOpTime"),
    "R2CTime" -> createNanoTimingMetric(MODERATE_LEVEL, "R2COpTime"),
    FILTER_TIME -> createNanoTimingMetric(DEBUG_LEVEL, DESCRIPTION_FILTER_TIME),
    "scanTime" -> createTimingMetric(ESSENTIAL_LEVEL, "scan time")
  )

  override protected def doExecute(): RDD[InternalRow] =
    throw new IllegalStateException(s"Row-based execution should not occur for $this")

  override protected def internalDoExecuteColumnar(): RDD[ColumnarBatch] = {
    val numOutputRows = gpuLongMetric(NUM_OUTPUT_ROWS)
    val scanTime = gpuLongMetric("scanTime")
    inputRDD.asInstanceOf[RDD[ColumnarBatch]].mapPartitionsInternal { batches =>
      new Iterator[ColumnarBatch] {

        override def hasNext: Boolean = {
          val startNs = System.nanoTime()
          val hasNext = batches.hasNext
          scanTime += NANOSECONDS.toMillis(System.nanoTime() - startNs)
          hasNext
        }

        override def next(): ColumnarBatch = {
          val startNs = System.nanoTime()
          val batch = batches.next()
          scanTime += NANOSECONDS.toMillis(System.nanoTime() - startNs)
          numOutputRows += batch.numRows()
          batch
        }
      }
    }
  }

  override def doCanonicalize(): VeloxFileSourceScanExec = {
    VeloxFileSourceScanExec(
      relation,
      originalOutput.map(QueryPlan.normalizeExpressions(_, originalOutput)),
      requiredSchema,
      QueryPlan.normalizePredicates(
        filterUnusedDynamicPruningExpressions(partitionFilters), originalOutput),
      optionalBucketSet,
      optionalNumCoalescedBuckets,
      QueryPlan.normalizePredicates(dataFilters, originalOutput),
      None,
      queryUsesInputFile,
      alluxioPathsMap = alluxioPathsMap)(rapidsConf)
  }

  // Filters unused DynamicPruningExpression expressions - one which has been replaced
  // with DynamicPruningExpression(Literal.TrueLiteral) during Physical Planning
  private def filterUnusedDynamicPruningExpressions(
                                                     predicates: Seq[Expression]): Seq[Expression] = {
    predicates.filterNot(_ == DynamicPruningExpression(Literal.TrueLiteral))
  }

  override def otherCopyArgs: Seq[AnyRef] = Seq(rapidsConf)

}
