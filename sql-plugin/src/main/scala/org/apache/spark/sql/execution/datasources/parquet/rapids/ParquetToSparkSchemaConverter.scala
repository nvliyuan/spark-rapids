/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.sql.execution.datasources.parquet.rapids

import java.util.Locale

import org.apache.hadoop.conf.Configuration
import org.apache.parquet.io.{ColumnIO, ColumnIOFactory, GroupColumnIO, PrimitiveColumnIO}
import org.apache.parquet.schema._
import org.apache.parquet.schema.LogicalTypeAnnotation._
import org.apache.parquet.schema.PrimitiveType.PrimitiveTypeName._
import org.apache.parquet.schema.Type.Repetition._

import org.apache.spark.sql.errors.QueryCompilationErrors
import org.apache.spark.sql.execution.datasources.parquet.ParquetSchemaConverter
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.types._

/**
 * This converter class is used to convert Parquet [[MessageType]] to Spark SQL [[StructType]]
 * (via the `convert` method) as well as [[ParquetColumn]] (via the `convertParquetColumn`
 * method). The latter contains richer information about the Parquet type, including its
 * associated repetition & definition level, column path, column descriptor etc.
 *
 * Parquet format backwards-compatibility rules are respected when converting Parquet
 * [[MessageType]] schemas.
 *
 * @see https://github.com/apache/parquet-format/blob/master/LogicalTypes.md
 *
 * @param assumeBinaryIsString Whether unannotated BINARY fields should be assumed to be Spark SQL
 *        [[StringType]] fields.
 * @param assumeInt96IsTimestamp Whether unannotated INT96 fields should be assumed to be Spark SQL
 *        [[TimestampType]] fields.
 * @param caseSensitive Whether use case sensitive analysis when comparing Spark catalyst read
 *                      schema with Parquet schema.
 * @param inferTimestampNTZ Whether TimestampNTZType type is enabled.
 * @param nanosAsLong Whether timestamps with nanos are converted to long.
 */
class ParquetToSparkSchemaConverter(
    assumeBinaryIsString: Boolean = SQLConf.PARQUET_BINARY_AS_STRING.defaultValue.get,
    assumeInt96IsTimestamp: Boolean = SQLConf.PARQUET_INT96_AS_TIMESTAMP.defaultValue.get,
    caseSensitive: Boolean = SQLConf.CASE_SENSITIVE.defaultValue.get,
    inferTimestampNTZ: Boolean = false,
    nanosAsLong: Boolean = false) {

  def this(conf: SQLConf) = this(
    assumeBinaryIsString = conf.isParquetBinaryAsString,
    assumeInt96IsTimestamp = conf.isParquetINT96AsTimestamp,
    caseSensitive = conf.caseSensitiveAnalysis)

  def this(conf: Configuration) = this(
    assumeBinaryIsString = conf.get(SQLConf.PARQUET_BINARY_AS_STRING.key).toBoolean,
    assumeInt96IsTimestamp = conf.get(SQLConf.PARQUET_INT96_AS_TIMESTAMP.key).toBoolean,
    caseSensitive = conf.get(SQLConf.CASE_SENSITIVE.key).toBoolean)

  /**
   * Returns true if TIMESTAMP_NTZ type is enabled in this ParquetToSparkSchemaConverter.
   */
  def isTimestampNTZEnabled(): Boolean = {
    inferTimestampNTZ
  }

  /**
   * Converts Parquet [[MessageType]] `parquetSchema` to a Spark SQL [[StructType]].
   */
  def convert(parquetSchema: MessageType): StructType = {
    val column = new ColumnIOFactory().getColumnIO(parquetSchema)
    val converted = convertInternal(column)
    converted.sparkType.asInstanceOf[StructType]
  }

  /**
   * Convert `parquetSchema` into a [[ParquetColumn]] which contains its corresponding Spark
   * SQL [[StructType]] along with other information such as the maximum repetition and definition
   * level of each node, column descriptor for the leave nodes, etc.
   *
   * If `sparkReadSchema` is not empty, when deriving Spark SQL type from a Parquet field this will
   * check if the same field also exists in the schema. If so, it will use the Spark SQL type.
   * This is necessary since conversion from Parquet to Spark could cause precision loss. For
   * instance, Spark read schema is smallint/tinyint but Parquet only support int.
   */
  def convertParquetColumn(
      parquetSchema: MessageType,
      sparkReadSchema: Option[StructType] = None): ParquetColumn = {
    val column = new ColumnIOFactory().getColumnIO(parquetSchema)
    convertInternal(column, sparkReadSchema)
  }

  private def convertInternal(
      groupColumn: GroupColumnIO,
      sparkReadSchema: Option[StructType] = None): ParquetColumn = {
    // First convert the read schema into a map from field name to the field itself, to avoid O(n)
    // lookup cost below.
    val schemaMapOpt = sparkReadSchema.map { schema =>
      schema.map(f => normalizeFieldName(f.name) -> f).toMap
    }

    val converted = (0 until groupColumn.getChildrenCount).map { i =>
      val field = groupColumn.getChild(i)
      val fieldFromReadSchema = schemaMapOpt.flatMap { schemaMap =>
        schemaMap.get(normalizeFieldName(field.getName))
      }
      var fieldReadType = fieldFromReadSchema.map(_.dataType)

      // If a field is repeated here then it is neither contained by a `LIST` nor `MAP`
      // annotated group (these should've been handled in `convertGroupField`), e.g.:
      //
      //  message schema {
      //    repeated int32 int_array;
      //  }
      // or
      //  message schema {
      //    repeated group struct_array {
      //      optional int32 field;
      //    }
      //  }
      //
      // the corresponding Spark read type should be an array and we should pass the element type
      // to the group or primitive type conversion method.
      if (field.getType.getRepetition == REPEATED) {
        fieldReadType = fieldReadType.flatMap {
          case at: ArrayType => Some(at.elementType)
          case _ =>
            throw QueryCompilationErrors.illegalParquetTypeError(groupColumn.toString)
        }
      }

      val convertedField = convertField(field, fieldReadType)
      val fieldName = fieldFromReadSchema.map(_.name).getOrElse(field.getType.getName)

      field.getType.getRepetition match {
        case OPTIONAL | REQUIRED =>
          val nullable = field.getType.getRepetition == OPTIONAL
          (StructField(fieldName, convertedField.sparkType, nullable = nullable),
              convertedField)

        case REPEATED =>
          // A repeated field that is neither contained by a `LIST`- or `MAP`-annotated group nor
          // annotated by `LIST` or `MAP` should be interpreted as a required list of required
          // elements where the element type is the type of the field.
          val arrayType = ArrayType(convertedField.sparkType, containsNull = false)
          (StructField(fieldName, arrayType, nullable = false),
              ParquetColumn(arrayType, None, convertedField.repetitionLevel - 1,
                convertedField.definitionLevel - 1, required = true, convertedField.path,
                Seq(convertedField.copy(required = true))))
      }
    }

    ParquetColumn(StructType(converted.map(_._1)), groupColumn, converted.map(_._2))
  }

  private def normalizeFieldName(name: String): String =
    if (caseSensitive) name else name.toLowerCase(Locale.ROOT)

  /**
   * Converts a Parquet [[Type]] to a [[ParquetColumn]] which wraps a Spark SQL [[DataType]] with
   * additional information such as the Parquet column's repetition & definition level, column
   * path, column descriptor etc.
   */
  def convertField(
      field: ColumnIO,
      sparkReadType: Option[DataType]): ParquetColumn = {
    val targetType = sparkReadType.map {
      case udt: UserDefinedType[_] => udt.sqlType
      case otherType => otherType
    }
    field match {
      case primitiveColumn: PrimitiveColumnIO => convertPrimitiveField(primitiveColumn, targetType)
      case groupColumn: GroupColumnIO => convertGroupField(groupColumn, targetType)
    }
  }

  private def convertPrimitiveField(
      primitiveColumn: PrimitiveColumnIO,
      sparkReadType: Option[DataType]): ParquetColumn = {
    val parquetType = primitiveColumn.getType.asPrimitiveType()
    val typeAnnotation = primitiveColumn.getType.getLogicalTypeAnnotation
    val typeName = primitiveColumn.getPrimitive

    def typeString =
      if (typeAnnotation == null) s"$typeName" else s"$typeName ($typeAnnotation)"

    def typeNotImplemented() =
      throw QueryCompilationErrors.parquetTypeUnsupportedYetError(typeString)

    def illegalType() =
      throw QueryCompilationErrors.illegalParquetTypeError(typeString)

    // When maxPrecision = -1, we skip precision range check, and always respect the precision
    // specified in field.getDecimalMetadata.  This is useful when interpreting decimal types stored
    // as binaries with variable lengths.
    def makeDecimalType(maxPrecision: Int = -1): DecimalType = {
      val decimalLogicalTypeAnnotation = typeAnnotation
        .asInstanceOf[DecimalLogicalTypeAnnotation]
      val precision = decimalLogicalTypeAnnotation.getPrecision
      val scale = decimalLogicalTypeAnnotation.getScale

      ParquetSchemaConverter.checkConversionRequirement(
        maxPrecision == -1 || 1 <= precision && precision <= maxPrecision,
        s"Invalid decimal precision: $typeName cannot store $precision digits (max $maxPrecision)")

      DecimalType(precision, scale)
    }

    val sparkType = sparkReadType.getOrElse(typeName match {
      case BOOLEAN => BooleanType

      case FLOAT => FloatType

      case DOUBLE => DoubleType

      case INT32 =>
        typeAnnotation match {
          case intTypeAnnotation: IntLogicalTypeAnnotation if intTypeAnnotation.isSigned =>
            intTypeAnnotation.getBitWidth match {
              case 8 => ByteType
              case 16 => ShortType
              case 32 => IntegerType
              case _ => illegalType()
            }
          case null => IntegerType
          case _: DateLogicalTypeAnnotation => DateType
          case _: DecimalLogicalTypeAnnotation => makeDecimalType(Decimal.MAX_INT_DIGITS)
          case intTypeAnnotation: IntLogicalTypeAnnotation if !intTypeAnnotation.isSigned =>
            intTypeAnnotation.getBitWidth match {
              case 8 => ShortType
              case 16 => IntegerType
              case 32 => LongType
              case _ => illegalType()
            }
          case t: TimestampLogicalTypeAnnotation if t.getUnit == TimeUnit.MILLIS =>
            typeNotImplemented()
          case _ => illegalType()
        }

      case INT64 =>
        typeAnnotation match {
          case intTypeAnnotation: IntLogicalTypeAnnotation if intTypeAnnotation.isSigned =>
            intTypeAnnotation.getBitWidth match {
              case 64 => LongType
              case _ => illegalType()
            }
          case null => LongType
          case _: DecimalLogicalTypeAnnotation => makeDecimalType(Decimal.MAX_LONG_DIGITS)
          case intTypeAnnotation: IntLogicalTypeAnnotation if !intTypeAnnotation.isSigned =>
            intTypeAnnotation.getBitWidth match {
              // The precision to hold the largest unsigned long is:
              // `java.lang.Long.toUnsignedString(-1).length` = 20
              case 64 => DecimalType(20, 0)
              case _ => illegalType()
            }
          case timestamp: TimestampLogicalTypeAnnotation
            if timestamp.getUnit == TimeUnit.MICROS || timestamp.getUnit == TimeUnit.MILLIS =>
            if (timestamp.isAdjustedToUTC || !inferTimestampNTZ) {
              TimestampType
            } else {
              TimestampNTZType
            }
          // SPARK-40819: NANOS are not supported as a Timestamp, convert to LongType without
          // timezone awareness to address behaviour regression introduced by SPARK-34661
          case timestamp: TimestampLogicalTypeAnnotation
            if timestamp.getUnit == TimeUnit.NANOS && nanosAsLong =>
            LongType
          case _ => illegalType()
        }

      case INT96 =>
        ParquetSchemaConverter.checkConversionRequirement(
          assumeInt96IsTimestamp,
          "INT96 is not supported unless it's interpreted as timestamp. " +
            s"Please try to set ${SQLConf.PARQUET_INT96_AS_TIMESTAMP.key} to true.")
        TimestampType

      case BINARY =>
        typeAnnotation match {
          case _: StringLogicalTypeAnnotation | _: EnumLogicalTypeAnnotation |
               _: JsonLogicalTypeAnnotation => StringType
          case null if assumeBinaryIsString => StringType
          case null => BinaryType
          case _: BsonLogicalTypeAnnotation => BinaryType
          case _: DecimalLogicalTypeAnnotation => makeDecimalType()
          case _ => illegalType()
        }

      case FIXED_LEN_BYTE_ARRAY =>
        typeAnnotation match {
          case _: DecimalLogicalTypeAnnotation =>
            makeDecimalType(Decimal.maxPrecisionForBytes(parquetType.getTypeLength))
          case _: IntervalLogicalTypeAnnotation => typeNotImplemented()
          case null => BinaryType
          case _ => illegalType()
        }

      case _ => illegalType()
    })

    ParquetColumn(sparkType, primitiveColumn)
  }

  private def convertGroupField(
      groupColumn: GroupColumnIO,
      sparkReadType: Option[DataType]): ParquetColumn = {
    val field = groupColumn.getType.asGroupType()
    Option(field.getLogicalTypeAnnotation).fold(
      convertInternal(groupColumn, sparkReadType.map(_.asInstanceOf[StructType]))) {
      // A Parquet list is represented as a 3-level structure:
      //
      //   <list-repetition> group <name> (LIST) {
      //     repeated group list {
      //       <element-repetition> <element-type> element;
      //     }
      //   }
      //
      // However, according to the most recent Parquet format spec (not released yet up until
      // writing), some 2-level structures are also recognized for backwards-compatibility.  Thus,
      // we need to check whether the 2nd level or the 3rd level refers to list element type.
      //
      // See: https://github.com/apache/parquet-format/blob/master/LogicalTypes.md#lists
      case _: ListLogicalTypeAnnotation =>
        ParquetSchemaConverter.checkConversionRequirement(
          field.getFieldCount == 1, s"Invalid list type $field")
        ParquetSchemaConverter.checkConversionRequirement(
          sparkReadType.forall(_.isInstanceOf[ArrayType]),
          s"Invalid Spark read type: expected $field to be list type but found $sparkReadType")

        val repeated = groupColumn.getChild(0)
        val repeatedType = repeated.getType
        ParquetSchemaConverter.checkConversionRequirement(
          repeatedType.isRepetition(REPEATED), s"Invalid list type $field")
        val sparkReadElementType = sparkReadType.map(_.asInstanceOf[ArrayType].elementType)

        if (isElementType(repeatedType, field.getName)) {
          var converted = convertField(repeated, sparkReadElementType)
          val convertedType = sparkReadElementType.getOrElse(converted.sparkType)

          // legacy format such as:
          //   optional group my_list (LIST) {
          //     repeated int32 element;
          //   }
          // we should mark the primitive field as required
          if (repeatedType.isPrimitive) converted = converted.copy(required = true)

          ParquetColumn(ArrayType(convertedType, containsNull = false),
            groupColumn, Seq(converted))
        } else {
          val element = repeated.asInstanceOf[GroupColumnIO].getChild(0)
          val converted = convertField(element, sparkReadElementType)
          val convertedType = sparkReadElementType.getOrElse(converted.sparkType)
          val optional = element.getType.isRepetition(OPTIONAL)
          ParquetColumn(ArrayType(convertedType, containsNull = optional),
            groupColumn, Seq(converted))
        }

      // scalastyle:off
      // `MAP_KEY_VALUE` is for backwards-compatibility
      // See: https://github.com/apache/parquet-format/blob/master/LogicalTypes.md#backward-compatibility-rules-1
      // scalastyle:on
      case _: MapLogicalTypeAnnotation | _: MapKeyValueTypeAnnotation =>
        ParquetSchemaConverter.checkConversionRequirement(
          field.getFieldCount == 1 && !field.getType(0).isPrimitive,
          s"Invalid map type: $field")
        ParquetSchemaConverter.checkConversionRequirement(
          sparkReadType.forall(_.isInstanceOf[MapType]),
          s"Invalid Spark read type: expected $field to be map type but found $sparkReadType")

        val keyValue = groupColumn.getChild(0).asInstanceOf[GroupColumnIO]
        val keyValueType = keyValue.getType.asGroupType()
        ParquetSchemaConverter.checkConversionRequirement(
          keyValueType.isRepetition(REPEATED) && keyValueType.getFieldCount == 2,
          s"Invalid map type: $field")

        val key = keyValue.getChild(0)
        val value = keyValue.getChild(1)
        val sparkReadKeyType = sparkReadType.map(_.asInstanceOf[MapType].keyType)
        val sparkReadValueType = sparkReadType.map(_.asInstanceOf[MapType].valueType)
        val convertedKey = convertField(key, sparkReadKeyType)
        val convertedValue = convertField(value, sparkReadValueType)
        val convertedKeyType = sparkReadKeyType.getOrElse(convertedKey.sparkType)
        val convertedValueType = sparkReadValueType.getOrElse(convertedValue.sparkType)
        val valueOptional = value.getType.isRepetition(OPTIONAL)
        ParquetColumn(
          MapType(convertedKeyType, convertedValueType,
            valueContainsNull = valueOptional),
          groupColumn, Seq(convertedKey, convertedValue))
      case _ =>
        throw QueryCompilationErrors.unrecognizedParquetTypeError(field.toString)
    }
  }

  // scalastyle:off
  // Here we implement Parquet LIST backwards-compatibility rules.
  // See: https://github.com/apache/parquet-format/blob/master/LogicalTypes.md#backward-compatibility-rules
  // scalastyle:on
  private[parquet] def isElementType(repeatedType: Type, parentName: String): Boolean = {
    {
      // For legacy 2-level list types with primitive element type, e.g.:
      //
      //    // ARRAY<INT> (nullable list, non-null elements)
      //    optional group my_list (LIST) {
      //      repeated int32 element;
      //    }
      //
      repeatedType.isPrimitive
    } || {
      // For legacy 2-level list types whose element type is a group type with 2 or more fields,
      // e.g.:
      //
      //    // ARRAY<STRUCT<str: STRING, num: INT>> (nullable list, non-null elements)
      //    optional group my_list (LIST) {
      //      repeated group element {
      //        required binary str (UTF8);
      //        required int32 num;
      //      };
      //    }
      //
      repeatedType.asGroupType().getFieldCount > 1
    } || {
      // For legacy 2-level list types generated by parquet-avro (Parquet version < 1.6.0), e.g.:
      //
      //    // ARRAY<STRUCT<str: STRING>> (nullable list, non-null elements)
      //    optional group my_list (LIST) {
      //      repeated group array {
      //        required binary str (UTF8);
      //      };
      //    }
      //
      repeatedType.getName == "array"
    } || {
      // For Parquet data generated by parquet-thrift, e.g.:
      //
      //    // ARRAY<STRUCT<str: STRING>> (nullable list, non-null elements)
      //    optional group my_list (LIST) {
      //      repeated group my_list_tuple {
      //        required binary str (UTF8);
      //      };
      //    }
      //
      repeatedType.getName == s"${parentName}_tuple"
    }
  }
}
