/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

import org.apache.spark.sql.connector.read.Scan

trait GpuBatchScanExecMetrics extends GpuExec {
  import GpuMetric._

  def scan: Scan

  override def supportsColumnar = true

  override val outputRowsLevel: MetricsLevel = ESSENTIAL_LEVEL
  override val outputBatchesLevel: MetricsLevel = MODERATE_LEVEL
  override lazy val additionalMetrics: Map[String, GpuMetric] = Map(
    GPU_DECODE_TIME -> createNanoTimingMetric(MODERATE_LEVEL, DESCRIPTION_GPU_DECODE_TIME),
    BUFFER_TIME -> createNanoTimingMetric(MODERATE_LEVEL, DESCRIPTION_BUFFER_TIME),
    FILTER_TIME -> createNanoTimingMetric(DEBUG_LEVEL, DESCRIPTION_FILTER_TIME)
  ) ++ fileCacheMetrics ++ hybridScanMetrics

  lazy val fileCacheMetrics: Map[String, GpuMetric] = {
    // File cache only supported on Parquet files for now.
    scan match {
      case _: GpuParquetScan | _: GpuOrcScan => createFileCacheMetrics()
      case _ => Map.empty
    }
  }

  lazy val hybridScanMetrics: Map[String, GpuMetric] = {
    scan match {
      case parquetScan: GpuParquetScan =>
        val b = scala.collection.mutable.ArrayBuffer[(String, GpuMetric)]()
        // Some additional metrics to inspect the performance issues of Parquet reading
        b += READ_FS_TIME -> createNanoTimingMetric(DEBUG_LEVEL, DESCRIPTION_READ_FS_TIME)
        b += WRITE_BUFFER_TIME -> createNanoTimingMetric(DEBUG_LEVEL, DESCRIPTION_WRITE_BUFFER_TIME)
        b += "filteredRowGroups" -> createMetric(DEBUG_LEVEL, "filtered row groups")
        b += "totalRowGroups" -> createMetric(DEBUG_LEVEL, "total row groups")
        b += "decodeGpuWait" -> createNanoTimingMetric(DEBUG_LEVEL, "GPU wait time for Decode")
        // Metrics for HybridParquetScan
        if (parquetScan.rapidsConf.parquetScanHybridMode != "GPU_ONLY") {
          b += "cpuDecodeBatches" -> createMetric(DEBUG_LEVEL, "CPU decode batches")
          b += "preloadH2DBatches" -> createMetric(DEBUG_LEVEL, "preload batches")
          b += "cpuDecodeRows" -> createMetric(DEBUG_LEVEL, "CPU decode rows")
          b += "cpuDecodeTime" -> createNanoTimingMetric(DEBUG_LEVEL, "CPU decode time")
          b += "hostVecToDeviceTime" -> createNanoTimingMetric(DEBUG_LEVEL,
            "host To device Time")
          b += "hybridPollTime" -> createNanoTimingMetric(DEBUG_LEVEL, "hybridPollTime")
          b += "waitAsyncDecode" -> createNanoTimingMetric(DEBUG_LEVEL,
            "waitTimeForAsyncCpuDecode")
          b += "preH2dGpuWait" -> createNanoTimingMetric(DEBUG_LEVEL,
            "GPU wait time before H2D")
        }
        b.toMap
      case _ => Map.empty
    }
  }
}
