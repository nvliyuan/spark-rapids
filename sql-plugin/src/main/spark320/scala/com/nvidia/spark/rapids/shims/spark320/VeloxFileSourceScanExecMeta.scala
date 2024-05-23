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

/*** spark-rapids-shim-json-lines
{"spark": "320"}
{"spark": "321"}
{"spark": "321cdh"}
{"spark": "322"}
{"spark": "323"}
{"spark": "324"}
spark-rapids-shim-json-lines ***/
package com.nvidia.spark.rapids.shims

import com.nvidia.spark.rapids._

import org.apache.spark.rapids.velox.VeloxFileSourceScanExec
import org.apache.spark.sql.catalyst.expressions.DynamicPruningExpression
import org.apache.spark.sql.execution._
import org.apache.spark.sql.execution.datasources.{HadoopFsRelation, InMemoryFileIndex}

class VeloxFileSourceScanExecMeta(plan: FileSourceScanExec,
                                  conf: RapidsConf,
                                  parent: Option[RapidsMeta[_, _, _]],
                                  rule: DataFromReplacementRule)
  extends SparkPlanMeta[FileSourceScanExec](plan, conf, parent, rule) {

  // Replaces SubqueryBroadcastExec inside dynamic pruning filters with GPU counterpart
  // if possible. Instead regarding filters as childExprs of current Meta, we create
  // a new meta for SubqueryBroadcastExec. The reason is that the GPU replacement of
  // FileSourceScan is independent from the replacement of the partitionFilters. It is
  // possible that the FileSourceScan is on the CPU, while the dynamic partitionFilters
  // are on the GPU. And vice versa.
  private lazy val partitionFilters = {
    val convertBroadcast = (bc: SubqueryBroadcastExec) => {
      val meta = GpuOverrides.wrapAndTagPlan(bc, conf)
      meta.tagForExplain()
      meta.convertIfNeeded().asInstanceOf[BaseSubqueryExec]
    }
    wrapped.partitionFilters.map { filter =>
      filter.transformDown {
        case dpe@DynamicPruningExpression(inSub: InSubqueryExec) =>
          inSub.plan match {
            case bc: SubqueryBroadcastExec =>
              dpe.copy(inSub.copy(plan = convertBroadcast(bc)))
            case reuse@ReusedSubqueryExec(bc: SubqueryBroadcastExec) =>
              dpe.copy(inSub.copy(plan = reuse.copy(convertBroadcast(bc))))
            case _ =>
              dpe
          }
      }
    }
  }

  // partition filters and data filters are not run on the GPU
  override val childExprs: Seq[ExprMeta[_]] = Seq.empty

  override def tagPlanForGpu(): Unit = ScanExecShims.tagGpuFileSourceScanExecSupport(this)

  override def convertToGpu(): GpuExec = {
    val sparkSession = wrapped.relation.sparkSession
    val options = wrapped.relation.options
    val (location, _) =
      if (AlluxioCfgUtils.enabledAlluxioReplacementAlgoConvertTime(conf)) {
        val shouldReadFromS3 = wrapped.relation.location match {
          // Only handle InMemoryFileIndex
          //
          // skip handle `MetadataLogFileIndex`, from the description of this class:
          // it's about the files generated by the `FileStreamSink`.
          // The streaming data source is not in our scope.
          //
          // For CatalogFileIndex and FileIndex of `delta` data source,
          // need more investigation.
          case inMemory: InMemoryFileIndex =>
            // List all the partitions to reduce overhead, pass in 2 empty filters.
            // Subsequent process will do the right partition pruning.
            // This operation is fast, because it lists files from the caches and the caches
            // already exist in the `InMemoryFileIndex`.
            val pds = inMemory.listFiles(Seq.empty, Seq.empty)
            AlluxioUtils.shouldReadDirectlyFromS3(conf, pds)
          case _ =>
            false
        }

        if (!shouldReadFromS3) {
          // it's convert time algorithm and some paths are not large tables
          AlluxioUtils.replacePathIfNeeded(
            conf,
            wrapped.relation,
            partitionFilters,
            wrapped.dataFilters)
        } else {
          // convert time algorithm and read large files
          (wrapped.relation.location, None)
        }
      } else {
        // it's not convert time algorithm or read large files, do not replace
        (wrapped.relation.location, None)
      }

    val newRelation = HadoopFsRelation(
      location,
      wrapped.relation.partitionSchema,
      wrapped.relation.dataSchema,
      wrapped.relation.bucketSpec,
      wrapped.relation.fileFormat,
      options)(sparkSession)

    VeloxFileSourceScanExec(
      newRelation,
      wrapped.output,
      wrapped.requiredSchema,
      partitionFilters,
      wrapped.optionalBucketSet,
      wrapped.optionalNumCoalescedBuckets,
      wrapped.dataFilters,
      wrapped.tableIdentifier,
      wrapped.disableBucketedScan,
      queryUsesInputFile = false,
      None)(conf)
  }
}
