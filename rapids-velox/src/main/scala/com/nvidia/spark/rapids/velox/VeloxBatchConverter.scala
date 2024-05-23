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

import scala.collection.mutable

import ai.rapids.cudf.{DType, HostColumnVector, HostColumnVectorCore, HostMemoryBuffer}
import ai.rapids.cudf.DType.DTypeEnum
import io.glutenproject.columnarbatch.IndicatorVector
import io.glutenproject.rapids.GlutenJniWrapper

import org.apache.spark.internal.Logging
import org.apache.spark.sql.execution.metric.SQLMetric
import org.apache.spark.sql.types._
import org.apache.spark.sql.vectorized.ColumnarBatch


case class ColumnAllocInfo(veloxType: VeloxDataTypes.Type,
                           readType: StructField,
                           numRows: Int,
                           nullCnt: Int,
                           offsetsSize: Int,
                           dataSize: Int,
                           children: Seq[ColumnAllocInfo])

class VeloxBatchConverter(rt: GlutenJniWrapper,
                          nativeHandle: Long,
                          schema: StructType,
                          metrics: Map[String, SQLMetric]) extends Logging {

  private var stackedBatches = 0
  private var needToFlush = false

  def hasStackedBatches: Boolean = stackedBatches > 0

  def readyToFlush: Boolean = needToFlush

  def tryAppendBatch(cb: ColumnarBatch): Boolean = {
    if (rt.appendBatch(nativeHandle, VeloxBatchConverter.getNativeBatchHandle(cb))) {
      stackedBatches += 1
      true
    } else {
      needToFlush = true
      false
    }
  }

  def flushAndConvert(): Array[HostColumnVector] = {
    // Collect the sizes of buffers required to hold the target batch for stacked velox batches.
    val nativeAllocInfo = rt.getBufferInfo(nativeHandle)
    // The second position holds the total bytes of RapidsBuffers to be filled.
    metrics("OutputSizeInBytes") += nativeAllocInfo(1)
    val allocInfo = VeloxBatchConverter.decodeNativeConvMeta(nativeAllocInfo, schema)

    // Allocate memory for HostBuffers from PinnedMemoryPool
    val bufferInfo = mutable.ArrayBuffer[Long]()
    bufferInfo.append(0L)
    val result = allocInfo.map { rootInfo =>
      val vecBuilder = createVectorBuilder(rootInfo,
        isRoot = true,
        seed = scala.util.Random.nextInt(10000)
      )
      vecBuilder.build(bufferInfo).asInstanceOf[HostColumnVector]
    }
    bufferInfo(0) = bufferInfo.length

    // Do the conversion
    val convertStart = System.nanoTime()
    rt.convert(nativeHandle, bufferInfo.toArray)
    metrics("VeloxC2CConvertTime") += System.nanoTime() - convertStart
    stackedBatches = 0
    needToFlush = false

    result
  }

  private def createVectorBuilder(info: ColumnAllocInfo,
                                  isRoot: Boolean, seed: Int): VectorBuilder = {

    require(VeloxDataTypes.canConvert(info.veloxType, info.readType.dataType),
      s"can NOT convert ${info.veloxType} to ${info.readType.dataType}")

    val childBuilders = info.children.map(ch => createVectorBuilder(ch, isRoot = false, seed))

    val nullBuffer = if (info.nullCnt >= 0) {
      // ColumnView.getValidityBufferSize
      val nullMaskBytes = {
        val actualBytes = (info.numRows + 7) >> 3
        ((actualBytes + 63) >> 6) << 6
      }
      Some(HostMemoryBuffer.allocate(nullMaskBytes))
    } else {
      None
    }
    val dataBuffer = if (info.dataSize > 0) {
      Some(HostMemoryBuffer.allocate(info.dataSize))
    } else {
      None
    }
    val offsetBuffer = if (info.offsetsSize > 0) {
      Some(HostMemoryBuffer.allocate(info.offsetsSize))
    } else {
      None
    }

    VectorBuilder(seed, isRoot,
      info.readType,
      info.numRows,
      info.nullCnt max 0,
      nullBuffer, dataBuffer, offsetBuffer,
      childBuilders)
  }

  private case class VectorBuilder(seed: Int,
                                   isRoot: Boolean,
                                   field: StructField,
                                   numRows: Int,
                                   nullCnt: Int,
                                   nullBuffer: Option[HostMemoryBuffer],
                                   dataBuffer: Option[HostMemoryBuffer],
                                   offsetBuffer: Option[HostMemoryBuffer],
                                   children: Seq[VectorBuilder]) {
    def build(bufferInfo: mutable.ArrayBuffer[Long]): HostColumnVectorCore = {
      // bufferPtrs is in pre-order
      val typeIdx = VeloxDataTypes.encodeSparkType(field.dataType).toLong
      bufferInfo.append(
        typeIdx,
        dataBuffer.map(_.getAddress).getOrElse(0L),
        dataBuffer.map(_.getLength).getOrElse(0L),
        nullBuffer.map(_.getAddress).getOrElse(0L),
        nullBuffer.map(_.getLength).getOrElse(0L),
        offsetBuffer.map(_.getAddress).getOrElse(0L),
        offsetBuffer.map(_.getLength).getOrElse(0L),
      )
      /*
      logInfo(
        s"[$seed] isRoot: $isRoot; field: $field; " +
          s"numRows: $numRows; nullCount: $nullCnt; " +
          s"dataBuffer: ${dataBuffer.map(_.getLength).getOrElse(0)}; " +
          s"nullBuffer: ${nullBuffer.map(_.getLength).getOrElse(0)}; " +
          s"offsetBuffer: ${offsetBuffer.map(_.getLength).getOrElse(0)}; " +
          s"numChildren: ${children.length}"
      )
      */
      var childVecs = new java.util.ArrayList[HostColumnVectorCore]()
      children.foreach(b => childVecs.add(b.build(bufferInfo)))

      val dType = VeloxBatchConverter.mapSparkTypeToDType(field.dataType)
      val nullCount = java.util.Optional.of(nullCnt.toLong.asInstanceOf[java.lang.Long])

      // Cast Map[child0, child1] => List[Struct[child0, child1]]
      if (field.dataType.isInstanceOf[MapType]) {
        val structCol = new HostColumnVectorCore(DType.STRUCT,
          children.head.numRows.toLong, java.util.Optional.of(0L),
          null, null, null, childVecs)
        childVecs = new java.util.ArrayList[HostColumnVectorCore]()
        childVecs.add(structCol)
      }

      if (isRoot) {
        new HostColumnVector(dType, numRows.toLong, nullCount,
          dataBuffer.orNull, nullBuffer.orNull, offsetBuffer.orNull,
          childVecs)
      } else {
        new HostColumnVectorCore(dType, numRows.toLong, nullCount,
          dataBuffer.orNull, nullBuffer.orNull, offsetBuffer.orNull,
          childVecs)
      }
    }
  }
}

object VeloxBatchConverter extends Logging {

  def apply(firstBatch: ColumnarBatch,
            targetBatchSize: Int,
            schema: StructType,
            metrics: Map[String, SQLMetric]): VeloxBatchConverter = {
    val runtime = GlutenJniWrapper.create()
    val nullableInfo = VeloxBatchConverter.encodeNullableInfo(schema)
    logDebug(s"nullableInfo: ${nullableInfo.mkString(" | ")}")
    val firstHandle = getNativeBatchHandle(firstBatch)
    val handle = runtime.buildCoalesceConverter(firstHandle, targetBatchSize, nullableInfo)
    new VeloxBatchConverter(runtime, handle, schema, metrics)
  }

  private def getNativeBatchHandle(cb: ColumnarBatch): Long = {
    cb.column(0) match {
      case indicator: IndicatorVector =>
        indicator.handle()
      case cv =>
        throw new IllegalArgumentException(
          s"Expecting IndicatorVector, but got ${cv.getClass}")
    }
  }

  private def decodeNativeConvMeta(meta: Array[Long],
                                   schema: StructType): Array[ColumnAllocInfo] = {
    case class DecodeHelper(
      var progress: Int,
      head: Int,
      bound: Int,
      parent: DecodeHelper,
      targetType: StructField,
      children: mutable.Queue[ColumnAllocInfo] = mutable.Queue[ColumnAllocInfo]()
    )

    val tupleSize = 6
    val headLength = 2
    val vectorSize = (meta.length - headLength) / tupleSize
    // nativeMetaPrettyPrint("getAllocSize", meta, headLength, tupleSize)

    val buildAllocInfo = (helper: DecodeHelper, children: Seq[ColumnAllocInfo]) => {
      val offset = helper.head * tupleSize + headLength

      ColumnAllocInfo(
        veloxType = VeloxDataTypes.decodeVeloxType(meta(offset + 1).toInt),
        readType = helper.targetType,
        numRows = meta(offset + 2).toInt,
        nullCnt = meta(offset + 3).toInt,
        offsetsSize = meta(offset + 4).toInt,
        dataSize = meta(offset + 5).toInt,
        children = children)
    }

    val stack = mutable.Stack[DecodeHelper]()
    val virtualRoot = DecodeHelper(0, -1, vectorSize, null,
      StructField("virtualRoot", schema, nullable = false)
    )
    stack.push(virtualRoot)

    while (stack.nonEmpty) {
      val cursor = stack.top
      assert(cursor.progress <= cursor.bound)
      if (cursor.progress == cursor.bound) {
        stack.pop()
        if (cursor.parent != null) {
          assert(cursor.parent.progress < cursor.bound)
          cursor.parent.progress = cursor.bound
          cursor.parent.children.enqueue(buildAllocInfo(cursor, cursor.children))
        }
      } else {
        val children = mutable.ArrayBuffer[DecodeHelper]()
        val childFields = mutable.Queue[StructField]()
        cursor.targetType.dataType match {
          case ArrayType(et, hasNull) =>
            childFields.enqueue(StructField("", et, hasNull))
          case MapType(kt, vt, hasNull) =>
            childFields.enqueue(StructField("", kt, nullable = false))
            childFields.enqueue(StructField("", vt, hasNull))
          case StructType(f) =>
            childFields.enqueue(f: _*)
        }
        var i = cursor.progress
        while (i < cursor.bound) {
          val rangeEnd = meta(i * tupleSize + headLength).toInt
          children += DecodeHelper(i + 1, i, rangeEnd, cursor, childFields.dequeue())
          i = rangeEnd
        }
        // Reverse the childIterator to ensure children being handled in the original order.
        // Otherwise, the update of progress will NOT work.
        children.reverseIterator.foreach(stack.push)
      }
    }

    virtualRoot.children.toArray
  }

  private def nativeMetaPrettyPrint(title: String,
                                    array: Array[Long], offset: Int, step: Int): Unit = {
    val sb = mutable.StringBuilder.newBuilder
    (offset until array.length by step).foreach { i =>
      sb.append(s"  (${(i - offset) / step + 1}) ")
      sb.append((i until i + step).map(array(_)).mkString(" | "))
      sb.append('\n')
    }
    logInfo(s"$title: \n${sb.toString()}")
  }

  private def mapSparkTypeToDType(dt: DataType): DType = dt match {
    case _: BooleanType => DType.BOOL8
    case _: ByteType => DType.INT8
    case _: ShortType => DType.INT16
    case _: IntegerType => DType.INT32
    case _: LongType => DType.INT64
    case _: FloatType => DType.FLOAT32
    case _: DoubleType => DType.FLOAT64
    case _: StringType => DType.STRING
    case _: DateType => DType.TIMESTAMP_DAYS
    case _: ArrayType => DType.LIST
    case _: MapType => DType.LIST
    case _: StructType => DType.STRUCT
    case d: DecimalType if DecimalType.is32BitDecimalType(d) =>
      DType.create(DTypeEnum.DECIMAL32, -d.scale)
    case d: DecimalType if DecimalType.is64BitDecimalType(d) =>
      DType.create(DTypeEnum.DECIMAL64, -d.scale)
    case d: DecimalType =>
      DType.create(DTypeEnum.DECIMAL128, -d.scale)
    case dt => throw new IllegalArgumentException(s"unexpected $dt")
  }

  private def encodeNullableInfo(root: StructType): Array[Int] = {
    val flattened = mutable.ArrayBuffer.empty[Int]
    val stack = mutable.Stack[StructField]()
    root.reverseIterator.foreach(stack.push)
    while (stack.nonEmpty) {
      val field = stack.pop()
      flattened += (if (field.nullable) 1 else 0)
      field.dataType match {
        case at: ArrayType =>
          stack.push(StructField("ArrayElem", at.elementType, nullable = at.containsNull))
        case mt: MapType =>
          stack.push(StructField("MapValue", mt.valueType, nullable = mt.valueContainsNull))
          stack.push(StructField("MapKey", mt.keyType, nullable = false))
        case st: StructType =>
          st.reverseIterator.foreach(stack.push)
        case _ =>
      }
    }
    flattened.toArray
  }

}
