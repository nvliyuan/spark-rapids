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

package com.nvidia.spark.rapids

import java.nio.ByteBuffer
import java.nio.channels.{Channels, WritableByteChannel}

import com.nvidia.spark.rapids.jni.Profiler
import org.apache.hadoop.fs.Path

import org.apache.spark.api.plugin.PluginContext
import org.apache.spark.internal.Logging
import org.apache.spark.util.SerializableConfiguration

object ProfilerManager extends Logging {
  private var writer: Option[ProfileWriter] = None

  def init(pluginCtx: PluginContext, conf: RapidsConf): Unit = {
    require(writer.isEmpty, "Already initialized")
    writer = conf.profilePath.flatMap { pathPrefix =>
      val executorId = pluginCtx.executorID()
      if (shouldProfile(executorId, conf)) {
        logInfo("Starting profiler")
        val w = new ProfileWriter(pluginCtx, pathPrefix)
        Profiler.init(w)
        Some(w)
      } else {
        None
      }
    }
  }

  def shutdown(): Unit = {
    writer.foreach { w =>
      Profiler.shutdown()
      w.close()
    }
    writer = None
  }

  private def shouldProfile(executorId: String, conf: RapidsConf): Boolean = {
    conf.profileExecutors.split(',').toSet.contains(executorId)
  }
}

class ProfileWriter(
    val pluginCtx: PluginContext,
    profilePathPrefix: String) extends Profiler.DataWriter {
  private val executorId = pluginCtx.executorID()
  private val outPath = getOutputPath(profilePathPrefix)
  private val out = openOutput()
  private var isClosed = false

  override def write(data: ByteBuffer): Unit = {
    while (data.hasRemaining()) {
      out.write(data)
    }
  }

  override def close(): Unit = {
    if (!isClosed) {
      isClosed = true
      out.close()
      pluginCtx.send(ProfileEndMsg(executorId, outPath.toString))
    }
  }

  private def getAppId: String = {
    val appId = pluginCtx.conf.get("spark.app.id", "")
    if (appId.isEmpty) {
      java.lang.management.ManagementFactory.getRuntimeMXBean.getName
    } else {
      appId
    }
  }

  private def getOutputPath(prefix: String): Path = {
    val parentDir = new Path(prefix)
    new Path(parentDir, s"rapids-profile-$getAppId-$executorId.bin")
  }

  private def openOutput(): WritableByteChannel = {
    val hadoopConf = pluginCtx.ask(ProfileStartMsg(executorId, outPath.toString))
      .asInstanceOf[SerializableConfiguration].value
    val fs = outPath.getFileSystem(hadoopConf)
    val outStream = fs.create(outPath, false)
    Channels.newChannel(outStream)
  }
}

trait ProfileMsg

case class ProfileStartMsg(executorId: String, path: String) extends ProfileMsg

case class ProfileEndMsg(executorId: String, path: String) extends ProfileMsg
