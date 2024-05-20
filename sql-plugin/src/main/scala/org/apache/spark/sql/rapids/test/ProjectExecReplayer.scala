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

package org.apache.spark.sql.rapids.test

import java.io.File
import java.nio.ByteBuffer
import java.nio.file.{Files, Paths}

import scala.reflect.ClassTag

import ai.rapids.cudf.Table
import com.nvidia.spark.rapids.{GpuColumnVector, GpuProjectExec, GpuTieredProject, NoopMetric}
import com.nvidia.spark.rapids.Arm.withResource
import org.apache.hadoop.fs.Path

import org.apache.spark.SparkEnv
import org.apache.spark.internal.Logging
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.DataType

/**
 * Replayer for dumped Project Exec runtime.
 * for how to dump, refer to dev doc `replay-exec.md`
 */
object ProjectExecReplayer extends Logging {
  private def deserializeObject[T: ClassTag](readPath: String): T = {
    val bytes = Files.readAllBytes(Paths.get(readPath))
    val buffer = ByteBuffer.wrap(bytes)
    SparkEnv.get.closureSerializer.newInstance().deserialize(buffer)
  }

  /**
   * Replay data dir should contains, e.g.::
   * - 1740344132_GpuTieredProject.meta
   * - 1740344132_cb_types.meta
   * - 1740344132_cb_data_0_744305176.parquet
   * Here 1740344132 is hashCode for a Project.
   * Only replay one Parquet in the replay path
   * @param args specify data path and hashCode, e.g.: /path/to/replay 1740344132
   */
  def main(args: Array[String]): Unit = {
    // check arguments and get paths
    if (args.length < 2) {
      logError("Project Exec replayer: Specify 2 args: data path and hashCode")
      return
    }
    var replayDir = args(0)
    val projectHash = args(1)
    logWarning("Project Exec replayer: start running.")

    // start a Spark session with Spark-Rapids initialization
    SparkSession.builder()
        .master("local[*]")
        .config("spark.plugins", "com.nvidia.spark.SQLPlugin")
        .appName("Test Baidu get_json_object diffs")
        .getOrCreate()
    logWarning("Project Exec replayer: started a Spark session")

    // copy to local dir
    replayDir = copyToLocal(replayDir)

    val cbTypesPath = replayDir + s"/${projectHash}_cb_types.meta"
    if (!(new File(cbTypesPath).exists() && new File(cbTypesPath).isFile)) {
      logError(s"Project Exec replayer: there is no cb_types.meta file in $replayDir")
      return
    }
    val projectMetaPath = replayDir + s"/${projectHash}_GpuTieredProject.meta"
    if (!(new File(projectMetaPath).exists() && new File(projectMetaPath).isFile)) {
      logError(s"Project Exec replayer: there is no GpuTieredProject.meta file in $replayDir")
      return
    }

    // find a Parquet file, e.g.: xxx_cb_data_101656570.parquet
    val parquets = new File(replayDir).listFiles(
      f => f.getName.startsWith(s"${projectHash}_cb_data_") &&
          f.getName.endsWith(".parquet"))
    if (parquets == null || parquets.isEmpty) {
      logError(s"Project Exec replayer: there is no cb_data_xxx.parquet file in $replayDir")
      return
    }
    // NOTE: only replay 1st parquet
    val cbPath = parquets(0).getAbsolutePath

    // restore project meta
    val restoredProject: GpuTieredProject = deserializeObject[GpuTieredProject](projectMetaPath)
    // print expressions in project
    restoredProject.exprTiers.foreach { exprs =>
      exprs.foreach { expr =>
        logWarning(s"Project Exec replayer: Project expression: ${expr.sql}")
      }
    }
    logWarning("Project Exec replayer: restored Project Exec meta")

    // restore column batch data
    val restoredCbTypes = deserializeObject[Array[DataType]](cbTypesPath)

    // replay
    withResource(Table.readParquet(new File(cbPath))) { restoredTable =>
      // this `restoredCb` will be closed in the `projectCb`
      val restoredCb = GpuColumnVector.from(restoredTable, restoredCbTypes)
      logWarning("Project Exec replayer: restored column batch data")
      logWarning("Project Exec replayer: begin to replay")
      // execute project
      withResource(GpuProjectExec.projectCb(restoredProject, restoredCb, NoopMetric)) { retCB =>
        logWarning(s"Project Exec replayer: project result has ${retCB.numRows()} rows.")
      }
      logWarning("Project Exec replayer: project replay completed successfully!!!")
    }
  }

  private def copyToLocal(replayDir: String): String = {
    // used to access remote path
    val hadoopConf = SparkSession.active.sparkContext.hadoopConfiguration
    // copy from remote to local
    val replayDirPath = new Path(replayDir)
    val fs = replayDirPath.getFileSystem(hadoopConf)

    if (!fs.getScheme.equals("file")) {
      // remote file system; copy to local dir
      val uuid = java.util.UUID.randomUUID.toString
      val localReplayDir = s"/tmp/replay-exec-tmp-$uuid"
      fs.copyToLocalFile(replayDirPath, new Path(s"/tmp/replay-exec-tmp-$uuid"))
      localReplayDir
    } else {
      // Remove the 'file:' prefix. e.g.: file:/a/b/c => /a/b/c
      replayDirPath.toUri.getPath
    }
  }
}
