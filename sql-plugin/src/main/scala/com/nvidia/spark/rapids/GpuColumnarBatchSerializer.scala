/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

import java.io._
import java.nio.ByteBuffer
import java.nio.channels.Channels

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

import ai.rapids.cudf.{HostColumnVector, HostMemoryBuffer, JCudfSerialization, NvtxColor, NvtxRange}
import com.nvidia.spark.rapids.Arm.{closeOnExcept, withResource}
import com.nvidia.spark.rapids.RapidsPluginImplicits._
import com.nvidia.spark.rapids.ScalableTaskCompletion.onTaskCompletion

import org.apache.spark.TaskContext
import org.apache.spark.serializer.{DeserializationStream, SerializationStream, Serializer, SerializerInstance}
import org.apache.spark.sql.types.{ArrayType, DataType, MapType, NullType, StructType}
import org.apache.spark.sql.vectorized.ColumnarBatch

class SerializedBatchIterator(headerSize: Int, hostDataBatchSize: Long, in: InputStream)
  extends Iterator[(Int, ColumnarBatch)] {
  private[this] val channel = Channels.newChannel(in)
  private[this] var nextHeader: Option[HostMemoryBuffer] = None
  private[this] var batchBuffer: Option[HostMemoryBuffer] = None
  private[this] var batchBufferOffset: Long = 0L
  private[this] var toBeReturned: Option[ColumnarBatch] = None
  private[this] var streamClosed: Boolean = false

  // Don't install the callback if in a unit test
  Option(TaskContext.get()).foreach { tc =>
    onTaskCompletion(tc) {
      nextHeader.foreach(_.safeClose())
      nextHeader = None
      batchBuffer.foreach(_.safeClose())
      batchBuffer = None
      toBeReturned.foreach(_.safeClose())
      toBeReturned = None
      if (!streamClosed) {
        channel.close()
        streamClosed = true
      }
    }
  }

  def tryReadNextHeader(): Option[Long] = {
    if (!streamClosed && nextHeader.isEmpty) {
      withResource(new NvtxRange("Read Header", NvtxColor.YELLOW)) { _ =>
        var atEOF = false
        val hmb = HostMemoryBuffer.allocate(headerSize, false)
        closeOnExcept(hmb) { _ =>
          val bb = hmb.asByteBuffer()
          while (!atEOF && bb.hasRemaining) {
            if (channel.read(bb) == -1) {
              if (bb.position != 0) {
                throw new EOFException("Unexpected EOF while reading batch header")
              } else {
                atEOF = true
              }
            }
          }
        }
        if (atEOF) {
          hmb.safeClose()
          channel.close()
          streamClosed = true
        } else {
          nextHeader = Some(hmb)
        }
      }
    }
    nextHeader.map(getDataLen)
  }

  private def tryReadNext(): Option[ColumnarBatch] = nextHeader.map { header =>
    withResource(new NvtxRange("Read Batch", NvtxColor.YELLOW)) { _ =>
      val numColumns = SerializedTableColumn.getNumColumns(header)
      val numRows = SerializedTableColumn.getNumRows(header)
      if (numColumns > 0) {
        val dataLen = getDataLen(header)
        val hmb: HostMemoryBuffer = if (dataLen > hostDataBatchSize) {
          // large batches get dedicated buffers
          HostMemoryBuffer.allocate(dataLen)
        } else {
          val buf = if (batchBuffer.isEmpty || batchBufferOffset + dataLen > hostDataBatchSize) {
            batchBuffer.foreach(_.safeClose())
            val hmb = HostMemoryBuffer.allocate(hostDataBatchSize)
            batchBuffer = Some(hmb)
            batchBufferOffset = 0
            hmb
          } else {
            batchBuffer.get
          }
          val hmb = buf.slice(batchBufferOffset, dataLen)
          batchBufferOffset += dataLen
          hmb
        }
        closeOnExcept(hmb) { _ =>
          new HostByteBufferIterator(hmb).foreach { bb =>
            while (bb.hasRemaining) {
              if (channel.read(bb) == -1) {
                throw new EOFException("Unexpected EOF while reading columnar data")
              }
            }
          }
          SerializedTableColumn.from(header, numColumns, numRows, hmb)
        }
      } else {
        SerializedTableColumn.from(header, numColumns, numRows)
      }
    }
  }

  override def hasNext: Boolean = {
    tryReadNextHeader()
    nextHeader.isDefined
  }

  override def next(): (Int, ColumnarBatch) = {
    if (toBeReturned.isEmpty) {
      tryReadNextHeader()
      toBeReturned = tryReadNext()
      if (nextHeader.isEmpty || toBeReturned.isEmpty) {
        throw new NoSuchElementException("Walked off of the end...")
      }
    }
    val ret = toBeReturned.get
    toBeReturned = None
    nextHeader = None
    (0, ret)
  }

  private def getDataLen(header: HostMemoryBuffer): Long = {
    // HACK: This knows too much about the JCudfSerialization format
    // read big-endian size as long at end of header
    val x = header.getLong(headerSize - 8)
    java.lang.Long.reverseBytes(x)
  }
}

/**
 * Serializer for serializing `ColumnarBatch`s for use during normal shuffle.
 *
 * The serialization write path takes the cudf `Table` that is described by the `ColumnarBatch`
 * and uses cudf APIs to serialize the data into a sequence of bytes on the host. The data is
 * returned to the Spark shuffle code where it is compressed by the CPU and written to disk.
 *
 * The serialization read path is notably different. The sequence of serialized bytes IS NOT
 * deserialized into a cudf `Table` but rather tracked in host memory by a `ColumnarBatch`
 * that contains a [[SerializedTableColumn]]. During query planning, each GPU columnar shuffle
 * exchange is followed by a [[GpuShuffleCoalesceExec]] that expects to receive only these
 * custom batches of [[SerializedTableColumn]]. [[GpuShuffleCoalesceExec]] coalesces the smaller
 * shuffle partitions into larger tables before placing them on the GPU for further processing.
 *
 * @note The RAPIDS shuffle does not use this code.
 */
class GpuColumnarBatchSerializer(
    schema: Array[DataType],
    hostDataBatchSize: Long,
    dataSize: GpuMetric)
    extends Serializer with Serializable {
  private val headerSize = computeHeaderSize(schema)

  override def newInstance(): SerializerInstance =
    new GpuColumnarBatchSerializerInstance(headerSize, hostDataBatchSize, dataSize)

  override def supportsRelocationOfSerializedObjects: Boolean = true

  private def computeHeaderSize(schema: Array[DataType]): Int = {
    // HACK: This is too tightly coupled with JCudfSerialization implementation
    // table header always has:
    // - 4-byte magic number
    // - 2-byte version number
    // - 4-byte column count
    // - 4-byte row count
    // - 8-byte data buffer length
    val tableHeader = 4 + 2 + 4 + 4 + 8;
    tableHeader + schema.map(computeHeaderSize).sum
  }

  private def computeHeaderSize(dt: DataType): Int = {
    // column header always has:
    // - 4-byte type ID
    // - 4-byte type scale
    // - 4-byte null count
    val size = 12
    val childrenSize = dt match {
      case s: StructType =>
        s.map(f => computeHeaderSize(f.dataType)).sum
      case a: ArrayType =>
        computeHeaderSize(a.elementType)
      case m: MapType =>
        computeHeaderSize(m.keyType) + computeHeaderSize(m.valueType)
      case _ => 0
    }
    size + childrenSize
  }
}

private class GpuColumnarBatchSerializerInstance(
    headerSize: Int,
    hostDataBatchSize: Long,
    dataSize: GpuMetric) extends SerializerInstance {

  override def serializeStream(out: OutputStream): SerializationStream = new SerializationStream {
    private[this] val dOut: DataOutputStream =
      new DataOutputStream(new BufferedOutputStream(out))

    override def writeValue[T: ClassTag](value: T): SerializationStream = {
      val batch = value.asInstanceOf[ColumnarBatch]
      val numColumns = batch.numCols()
      val columns: Array[HostColumnVector] = new Array(numColumns)
      val toClose = new ArrayBuffer[AutoCloseable]()
      try {
        var startRow = 0
        val numRows = batch.numRows()
        if (batch.numCols() > 0) {
          val firstCol = batch.column(0)
          if (firstCol.isInstanceOf[SlicedGpuColumnVector]) {
            // We don't have control over ColumnarBatch to put in the slice, so we have to do it
            // for each column.  In this case we are using the first column.
            startRow = firstCol.asInstanceOf[SlicedGpuColumnVector].getStart
            for (i <- 0 until numColumns) {
              columns(i) = batch.column(i).asInstanceOf[SlicedGpuColumnVector].getBase
            }
          } else {
            for (i <- 0 until numColumns) {
              batch.column(i) match {
                case gpu: GpuColumnVector =>
                  val cpu = gpu.copyToHost()
                  toClose += cpu
                  columns(i) = cpu.getBase
                case cpu: RapidsHostColumnVector =>
                  columns(i) = cpu.getBase
              }
            }
          }

          dataSize += JCudfSerialization.getSerializedSizeInBytes(columns, startRow, numRows)
          val range = new NvtxRange("Serialize Batch", NvtxColor.YELLOW)
          try {
            JCudfSerialization.writeToStream(columns, dOut, startRow, numRows)
          } finally {
            range.close()
          }
        } else {
          val range = new NvtxRange("Serialize Row Only Batch", NvtxColor.YELLOW)
          try {
            JCudfSerialization.writeRowsToStream(dOut, numRows)
          } finally {
            range.close()
          }
        }
      } finally {
        toClose.safeClose()
      }
      this
    }

    override def writeKey[T: ClassTag](key: T): SerializationStream = {
      // The key is only needed on the map side when computing partition ids. It does not need to
      // be shuffled.
      assert(null == key || key.isInstanceOf[Int])
      this
    }

    override def writeAll[T: ClassTag](iter: Iterator[T]): SerializationStream = {
      // This method is never called by shuffle code.
      throw new UnsupportedOperationException
    }

    override def writeObject[T: ClassTag](t: T): SerializationStream = {
      // This method is never called by shuffle code.
      throw new UnsupportedOperationException
    }

    override def flush(): Unit = {
      dOut.flush()
    }

    override def close(): Unit = {
      dOut.close()
    }
  }


  override def deserializeStream(in: InputStream): DeserializationStream = {
    new DeserializationStream {
      override def asKeyValueIterator: Iterator[(Int, ColumnarBatch)] = {
        new SerializedBatchIterator(headerSize, hostDataBatchSize, in)
      }

      override def asIterator: Iterator[Any] = {
        // This method is never called by shuffle code.
        throw new UnsupportedOperationException
      }

      override def readKey[T]()(implicit classType: ClassTag[T]): T = {
        // We skipped serialization of the key in writeKey(), so just return a dummy value since
        // this is going to be discarded anyways.
        null.asInstanceOf[T]
      }

      override def readValue[T]()(implicit classType: ClassTag[T]): T = {
        // This method should never be called by shuffle code.
        throw new UnsupportedOperationException
      }

      override def readObject[T]()(implicit classType: ClassTag[T]): T = {
        // This method is never called by shuffle code.
        throw new UnsupportedOperationException
      }

      override def close(): Unit = {
        in.close()
      }
    }
  }

  // These methods are never called by shuffle code.
  override def serialize[T: ClassTag](t: T): ByteBuffer = throw new UnsupportedOperationException
  override def deserialize[T: ClassTag](bytes: ByteBuffer): T =
    throw new UnsupportedOperationException
  override def deserialize[T: ClassTag](bytes: ByteBuffer, loader: ClassLoader): T =
    throw new UnsupportedOperationException
}

/**
 * A special `ColumnVector` that describes a serialized table read from shuffle.
 * This appears in a `ColumnarBatch` to pass serialized tables to [[GpuShuffleCoalesceExec]]
 * which should always appear in the query plan immediately after a shuffle.
 */
class SerializedTableColumn(
    val header: HostMemoryBuffer,
    val numColumns: Int,
    val numRows: Int,
    val hostBuffer: HostMemoryBuffer) extends GpuColumnVectorBase(NullType) {
  override def close(): Unit = {
    header.safeClose()
    hostBuffer.safeClose()
  }

  override def hasNull: Boolean = throw new IllegalStateException("should not be called")

  override def numNulls(): Int = throw new IllegalStateException("should not be called")
}

object SerializedTableColumn {
  /**
   * Build a `ColumnarBatch` consisting of a single [[SerializedTableColumn]] describing
   * the specified serialized table.
   *
   * @param header header data for the serialized table
   * @return columnar batch to be passed to [[GpuShuffleCoalesceExec]]
   */
  def from(header: HostMemoryBuffer): ColumnarBatch = {
    val numColumns = getNumColumns(header)
    val numRows = getNumRows(header)
    from(header, numColumns, numRows)
  }

  /**
   * Build a `ColumnarBatch` consisting of a single [[SerializedTableColumn]] describing
   * the specified serialized table.
   *
   * @param header header data for the serialized table
   * @param data the table data
   * @return columnar batch to be passed to [[GpuShuffleCoalesceExec]]
   */
  def from(header: HostMemoryBuffer, data: HostMemoryBuffer): ColumnarBatch = {
    val numColumns = getNumColumns(header)
    val numRows = getNumRows(header)
    from(header, numColumns, numRows, data)
  }

  /**
   * Build a `ColumnarBatch` consisting of a single [[SerializedTableColumn]] describing
   * the specified serialized table.
   *
   * @param header header data for the serialized table
   * @param numColumns number of columns in the table
   * @param numRows number of rows in the table
   * @return columnar batch to be passed to [[GpuShuffleCoalesceExec]]
   */
  def from(header: HostMemoryBuffer, numColumns: Int, numRows: Int): ColumnarBatch = {
    from(header, numColumns, numRows, null)
  }

  /**
   * Build a `ColumnarBatch` consisting of a single [[SerializedTableColumn]] describing
   * the specified serialized table.
   *
   * @param header header data for the serialized table
   * @param numColumns number of columns in the table
   * @param numRows number of rows in the table
   * @param data the table data
   * @return columnar batch to be passed to [[GpuShuffleCoalesceExec]]
   */
  def from(
      header: HostMemoryBuffer,
      numColumns: Int,
      numRows: Int,
      data: HostMemoryBuffer = null): ColumnarBatch = {
    val column = new SerializedTableColumn(header, numColumns, numRows, data)
    new ColumnarBatch(Array(column), numRows)
  }

  def getMemoryUsed(batch: ColumnarBatch): Long = {
    var sum: Long = 0
    if (batch.numCols == 1) {
      val cv = batch.column(0)
      cv match {
        case serializedTableColumn: SerializedTableColumn
            if serializedTableColumn.hostBuffer != null =>
          sum += serializedTableColumn.hostBuffer.getLength
        case _ =>
      }
    }
    sum
  }

  def getNumColumns(header: HostMemoryBuffer): Int = {
    // HACK: This knows too much about the JCudfSerialization format
    // read big-endian int from table header for number of columns
    val x = header.getInt(6)
    java.lang.Integer.reverseBytes(x)
  }

  def getNumRows(header: HostMemoryBuffer): Int = {
    // HACK: This knows too much about the JCudfSerialization format
    // read big-endian int from table header for number of rows
    val x = header.getInt(10)
    java.lang.Integer.reverseBytes(x)
  }
}
