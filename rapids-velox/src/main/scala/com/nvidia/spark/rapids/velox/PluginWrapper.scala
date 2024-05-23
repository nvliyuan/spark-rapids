/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

package com.nvidia.spark.rapids.velox

import io.glutenproject.{GlutenConfig, GlutenPlugin}
import io.glutenproject.rapids.GlutenJniWrapper

import org.apache.spark.{SparkContext, TaskFailedReason}
import org.apache.spark.api.plugin.{DriverPlugin, ExecutorPlugin, PluginContext, SparkPlugin}
import org.apache.spark.internal.Logging
import org.apache.spark.sql.internal.StaticSQLConf


class PluginWrapper extends SparkPlugin {

  private val glutenPlugin = new GlutenPlugin()

  override def driverPlugin(): DriverPlugin = {
    new DriverPluginWrapper(glutenPlugin.driverPlugin())
  }

  override def executorPlugin(): ExecutorPlugin = {
    new ExecutorPluginWrapper(glutenPlugin.executorPlugin())
  }
}

class DriverPluginWrapper(wrapped: DriverPlugin)
  extends DriverPlugin with Logging {

  override def init(sc: SparkContext,
                    pluginContext: PluginContext): java.util.Map[String, String] = {
    val conf = pluginContext.conf()
    if (!conf.getBoolean(PluginWrapper.LOAD_VELOX_KEY, defaultValue = false)) {
      return new java.util.HashMap[String, String]()
    }

    conf.set(GlutenConfig.GLUTEN_ENABLE_KEY, "false")
    val ret = wrapped.init(sc, pluginContext)
    conf.set(
      StaticSQLConf.SPARK_SESSION_EXTENSIONS.key,
      conf.get(StaticSQLConf.SPARK_SESSION_EXTENSIONS.key)
        .split(",")
        .filter(_ != PluginWrapper.GLUTEN_SESSION_EXTENSION_NAME)
        .mkString(",")
    )
    ret
  }

  override def registerMetrics(appId: String, pluginContext: PluginContext): Unit = {
    wrapped.registerMetrics(appId, pluginContext)
  }

  override def shutdown(): Unit = {
    wrapped.shutdown()
  }
}

class ExecutorPluginWrapper(wrapped: ExecutorPlugin)
  extends ExecutorPlugin with Logging {

  override def init(ctx: PluginContext, extraConf: java.util.Map[String, String]): Unit = {
    val conf = ctx.conf()
    if (conf.getBoolean(PluginWrapper.LOAD_VELOX_KEY, defaultValue = false)) {
      conf.set(GlutenConfig.GLUTEN_ENABLE_KEY, "false")
      wrapped.init(ctx, extraConf)
    }
  }

  override def shutdown(): Unit = {
    wrapped.shutdown()
  }

  override def onTaskStart(): Unit = {
    wrapped.onTaskStart()
  }

  override def onTaskSucceeded(): Unit = {
    wrapped.onTaskSucceeded()
  }

  override def onTaskFailed(failureReason: TaskFailedReason): Unit = {
    wrapped.onTaskFailed(failureReason)
  }
}

object PluginWrapper {
  private[velox] val GLUTEN_SESSION_EXTENSION_NAME = "io.glutenproject.GlutenSessionExtensions"
  private[velox] val LOAD_VELOX_KEY = "spark.rapids.sql.loadVelox"
}
