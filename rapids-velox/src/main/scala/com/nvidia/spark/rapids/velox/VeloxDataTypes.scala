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

import org.apache.spark.sql.types._

object VeloxDataTypes extends Enumeration {
  type Type = Value

  val BOOLEAN, TINYINT, SMALLINT, INTEGER, BIGINT, REAL, DOUBLE,
  VARCHAR, VARBINARY, TIMESTAMP, HUGEINT, DATE,
  ARRAY, MAP, ROW, UNKNOWN, FUNCTION, OPAQUE, INVALID = Value

  def decodeVeloxType(idx: Int): Type = idx match {
    case 0 => BOOLEAN
    case 1 => TINYINT
    case 2 => SMALLINT
    case 3 => INTEGER
    case 4 => BIGINT
    case 5 => REAL
    case 6 => DOUBLE
    case 7 => VARCHAR
    case 8 => VARBINARY
    case 9 => TIMESTAMP
    case 10 => HUGEINT
    case 11 => DATE
    case 30 => ARRAY
    case 31 => MAP
    case 32 => ROW
    case 33 => UNKNOWN
    case 34 => FUNCTION
    case 35 => OPAQUE
    case 36 => INVALID
    case _ =>
      throw new IllegalArgumentException(s"Invalid $idx for VeloxDataType")
  }

  def canConvert(src: Type, dst: DataType): Boolean = {
    src match {
      case BOOLEAN => dst.isInstanceOf[BooleanType]
      case TINYINT => dst.isInstanceOf[ByteType]
      case SMALLINT => dst.isInstanceOf[ShortType]
      case INTEGER => dst.isInstanceOf[IntegerType] || dst.isInstanceOf[DateType]
      case BIGINT => dst match {
        case _: LongType => true
        // velox stores 32/64 bits decimal as long
        case d: DecimalType =>
          DecimalType.is32BitDecimalType(d) || DecimalType.is64BitDecimalType(d)
        case _ => false
      }
      case REAL => dst.isInstanceOf[FloatType]
      case DOUBLE => dst.isInstanceOf[DoubleType]
      case VARCHAR | VARBINARY => dst.isInstanceOf[StringType]
      case HUGEINT => dst match {
        // velox stores 128 bits decimal as huge int
        case d: DecimalType =>
          !DecimalType.is32BitDecimalType(d) && !DecimalType.is64BitDecimalType(d)
        case _ => false
      }
      case ARRAY => dst.isInstanceOf[ArrayType]
      case MAP => dst.isInstanceOf[MapType]
      case ROW => dst.isInstanceOf[StructType]
      case _ => false
    }
  }

  def encodeSparkType(dataType: DataType): Int = dataType match {
    case BooleanType => 1
    case ByteType => 2
    case ShortType => 3
    case IntegerType => 4
    case LongType => 5
    case FloatType => 6
    case DoubleType => 7
    case d: DecimalType if DecimalType.is32BitDecimalType(d) => 8
    case d: DecimalType if DecimalType.is64BitDecimalType(d) => 9
    case _: DecimalType => 10
    case StringType => 11
    case DateType => 12
    case _: ArrayType => 101
    case _: MapType => 102
    case _: StructType => 103
    case _ =>
      throw new IllegalArgumentException(s"Unsupported SparkType($dataType)")
  }
}
