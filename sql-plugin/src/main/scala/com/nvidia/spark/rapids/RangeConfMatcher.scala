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

import scala.util.Try

/**
 * Determines if a value is in a comma-separated list of values and/or
 * hyphenated ranges provided by the user for a configuration setting.
 */
class RangeConfMatcher(conf: RapidsConf, entry: ConfEntry[String]) {
  private val (stringSet, intRanges) = {
    val confVal = conf.get(entry)
    val parts = confVal.split(',')
    val (rangeParts, singleParts) = parts.partition(_.contains('-'))
    val ranges = try {
      rangeParts.map(RangeConfMatcher.parseRange)
    } catch {
      case e: IllegalArgumentException =>
        throw new IllegalArgumentException(s"Invalid range settings for $entry: $confVal", e)
    }
    (singleParts.map(_.trim).toSet, ranges)
  }

  /** Returns true if the string value is in the configured values or ranges. */
  def contains(v: String): Boolean = {
    stringSet.contains(v) || (intRanges.nonEmpty && Try(v.toInt).map(checkRanges).getOrElse(false))
  }

  /** Returns true if the integer value is in the configured values or ranges. */
  def contains(v: Int): Boolean = {
    checkRanges(v) || stringSet.contains(v.toString)
  }

  private def checkRanges(v: Int): Boolean = {
    intRanges.exists {
      case (start, end) => start <= v && v <= end
    }
  }
}

object RangeConfMatcher {
  def parseRange(rangeStr: String): (Int,Int) = {
    val rangePair = rangeStr.split('-')
    if (rangePair.length != 2) {
      throw new IllegalArgumentException(s"Invalid range: $rangeStr")
    }
    val start = rangePair.head.trim.toInt
    val end = rangePair.last.trim.toInt
    if (end < start) {
      throw new IllegalArgumentException(s"Invalid range: $rangeStr")
    }
    (start, end)
  }
}