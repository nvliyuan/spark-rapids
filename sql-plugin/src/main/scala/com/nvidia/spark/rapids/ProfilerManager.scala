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
import java.util.{Timer, TimerTask}
import java.util.concurrent.TimeUnit

import com.nvidia.spark.rapids.jni.Profiler
import org.apache.hadoop.fs.Path

import org.apache.spark.api.plugin.PluginContext
import org.apache.spark.internal.Logging
import org.apache.spark.io.CompressionCodec
import org.apache.spark.sql.rapids.execution.TrampolineUtil
import org.apache.spark.util.SerializableConfiguration

object ProfilerManager extends Logging {
  private var writer: Option[ProfileWriter] = None
  private var timeRanges: Option[Seq[(Long, Long)]] = None
  private var timer: Option[Timer] = None
  private val startTimestamp = System.nanoTime()
  private var isProfileActive = false

  def init(pluginCtx: PluginContext, conf: RapidsConf): Unit = {
    require(writer.isEmpty, "Already initialized")
    timeRanges = conf.profileTimeRangesSeconds.map(parseTimeRanges)
    writer = conf.profilePath.flatMap { pathPrefix =>
      val executorId = pluginCtx.executorID()
      if (shouldProfile(executorId, conf)) {
        logInfo("Initializing profiler")
        val codec = conf.profileCompression match {
          case "none" => None
          case c => Some(TrampolineUtil.createCodec(pluginCtx.conf(), c))
        }
        val w = new ProfileWriter(pluginCtx, pathPrefix, codec)
        Profiler.init(w, conf.profileWriteBufferSize, conf.profileFlushPeriodMillis)
        Some(w)
      } else {
        None
      }
    }
    writer.foreach { _ =>
      updateAndSchedule()
    }
  }

  private def enable(): Unit = {
    writer.foreach { w =>
      if (!isProfileActive) {
        Profiler.start()
        isProfileActive = true
        w.pluginCtx.send(ProfileStatusMsg(w.executorId, "profile started"))
      }
    }
  }

  private def disable(): Unit = {
    writer.foreach { w =>
      if (isProfileActive) {
        Profiler.stop()
        isProfileActive = false
        w.pluginCtx.send(ProfileStatusMsg(w.executorId, "profile stopped"))
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
    val matcher = new RangeConfMatcher(conf, RapidsConf.PROFILE_EXECUTORS)
    matcher.contains(executorId)
  }

  private def parseTimeRanges(confVal: String): Seq[(Long, Long)] = {
    val ranges = try {
      confVal.split(',').map(RangeConfMatcher.parseRange).map {
        case (start, end) =>
          // convert relative time in seconds to absolute time in nanoseconds
          (startTimestamp + TimeUnit.SECONDS.toNanos(start),
            startTimestamp + TimeUnit.SECONDS.toNanos(end))
      }
    } catch {
      case e: IllegalArgumentException =>
        throw new IllegalArgumentException(
          s"Invalid range settings for ${RapidsConf.PROFILE_TIME_RANGES_SECONDS}: $confVal", e)
    }
    ranges.sorted.toIndexedSeq
  }

  private def updateAndSchedule(): Unit = {
    if (timeRanges.isDefined) {
      if (timer.isEmpty) {
        timer = Some(new Timer("profiler timer", true))
      }
      val now = System.nanoTime()
      // skip time ranges that have already passed
      val currentRanges = timeRanges.get.dropWhile {
        case (_, end) => end <= now
      }
      timeRanges = Some(currentRanges)
      if (currentRanges.isEmpty) {
        logWarning("No further time ranges to profile, shutting down")
        shutdown()
      } else {
        currentRanges.headOption.foreach {
          case (start, end) =>
            val timerDelay = if (start <= now) {
              enable()
              TimeUnit.NANOSECONDS.toMillis(end - now)
            } else {
              disable()
              TimeUnit.NANOSECONDS.toMillis(start - now)
            }
            timer.get.schedule(new TimerTask {
              override def run(): Unit = updateAndSchedule()
            }, timerDelay)
        }
      }
    } else {
      enable()
    }
  }
}

class ProfileWriter(
    val pluginCtx: PluginContext,
    profilePathPrefix: String,
    codec: Option[CompressionCodec]) extends Profiler.DataWriter {
  val executorId: String = pluginCtx.executorID()
  private val outPath = getOutputPath(profilePathPrefix, codec)
  private val out = openOutput(codec)
  private var isClosed = false

  override def write(data: ByteBuffer): Unit = {
    if (!isClosed) {
      while (data.hasRemaining) {
        out.write(data)
      }
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

  private def getOutputPath(prefix: String, codec: Option[CompressionCodec]): Path = {
    val parentDir = new Path(prefix)
    val suffix = codec.map(c => "." + TrampolineUtil.getCodecShortName(c.getClass.getName))
      .getOrElse("")
    new Path(parentDir, s"rapids-profile-$getAppId-$executorId.bin$suffix")
  }

  private def openOutput(codec: Option[CompressionCodec]): WritableByteChannel = {
    val hadoopConf = pluginCtx.ask(ProfileInitMsg(executorId, outPath.toString))
      .asInstanceOf[SerializableConfiguration].value
    val fs = outPath.getFileSystem(hadoopConf)
    val fsStream = fs.create(outPath, false)
    val outStream = codec.map(_.compressedOutputStream(fsStream)).getOrElse(fsStream)
    Channels.newChannel(outStream)
  }
}

trait ProfileMsg

case class ProfileInitMsg(executorId: String, path: String) extends ProfileMsg

case class ProfileStatusMsg(executorId: String, msg: String) extends ProfileMsg

case class ProfileEndMsg(executorId: String, path: String) extends ProfileMsg
