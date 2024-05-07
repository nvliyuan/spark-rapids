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

import java.io.{DataInputStream, DataOutputStream, EOFException}
import java.nio.ByteBuffer

import ai.rapids.cudf.{DeviceMemoryBuffer, HostMemoryBuffer, NvtxColor, NvtxRange}
import com.nvidia.spark.rapids.Arm.{closeOnExcept, withResource}
import com.nvidia.spark.rapids.ScalableTaskCompletion.onTaskCompletion
import com.nvidia.spark.rapids.format.TableMeta

import org.apache.spark.TaskContext
import org.apache.spark.sql.types.DataType
import org.apache.spark.sql.vectorized.ColumnarBatch

private sealed trait TableSerde extends AutoCloseable {
  protected val P_MAGIC_NUM: Int = 0x43554447 // "CUDF" + 1
  protected val P_VERSION: Int = 0
  protected val headerLen = 8 // the size in bytes of two Ints for a header

  // buffers for reuse, so it is should be only one instance of this trait per thread.
  protected val tmpBuf = new Array[Byte](1024 * 64) // 64k
  protected var hostBuffer: HostMemoryBuffer = _

  protected def getHostBuffer(len: Long): HostMemoryBuffer = {
    assert(len >= 0)
    if (hostBuffer != null && len <= hostBuffer.getLength) {
      hostBuffer.slice(0, len)
    } else { // hostBuffer is null or len is larger than the current one
      if (hostBuffer != null) {
        hostBuffer.close()
      }
      hostBuffer = HostMemoryBuffer.allocate(len)
      hostBuffer.slice(0, len)
    }
  }

  override def close(): Unit = {
    if (hostBuffer != null) {
      hostBuffer.close()
      hostBuffer = null
    }
  }
}

private[rapids] class SimpleTableSerializer extends TableSerde {
  private def writeByteBufferToStream(bBuf: ByteBuffer, dOut: DataOutputStream): Unit = {
    // Write the buffer size first
    val bufLen = bBuf.capacity()
    dOut.writeLong(bufLen.toLong)
    if (bBuf.hasArray) {
      dOut.write(bBuf.array())
    } else { // Probably a direct buffer
      var leftLen = bufLen
      while (leftLen > 0) {
        val copyLen = Math.min(tmpBuf.length, leftLen)
        bBuf.get(tmpBuf, 0, copyLen)
        dOut.write(tmpBuf, 0, copyLen)
        leftLen -= copyLen
      }
    }
  }

  private def writeHostBufferToStream(hBuf: HostMemoryBuffer, dOut: DataOutputStream): Unit = {
    // Write the buffer size first
    val bufLen = hBuf.getLength
    dOut.writeLong(bufLen)
    var leftLen = bufLen
    var hOffset = 0L
    while (leftLen > 0L) {
      val copyLen = Math.min(tmpBuf.length, leftLen)
      hBuf.getBytes(tmpBuf, 0, hOffset, copyLen)
      dOut.write(tmpBuf, 0, copyLen.toInt)
      leftLen -= copyLen
      hOffset += copyLen
    }
  }

  private def writeProtocolHeader(dOut: DataOutputStream): Unit = {
    dOut.writeInt(P_MAGIC_NUM)
    dOut.writeInt(P_VERSION)
  }

  def writeRowsOnlyToStream(numRows: Int, dOut: DataOutputStream): Long = {
    withResource(new NvtxRange("Serialize Rows Only Table", NvtxColor.RED)) { _ =>
      val degenBatch = new ColumnarBatch(Array.empty, numRows)
      val tableMetaBuf = MetaUtils.buildDegenerateTableMeta(degenBatch).getByteBuffer
      // 1) header, 2) metadata for an empty batch
      writeProtocolHeader(dOut)
      writeByteBufferToStream(tableMetaBuf, dOut)
      headerLen + tableMetaBuf.capacity()
    }
  }

  def writeToStream(hostTbl: PackedTableHostColumnVector, dOut: DataOutputStream): Long = {
    withResource(new NvtxRange("Serialize Host Table", NvtxColor.RED)) { _ =>
      // In the order of 1) header, 2) table metadata, 3) table data on host
      val metaBuf = hostTbl.getTableMeta.getByteBuffer
      val dataBuf = hostTbl.getTableBuffer
      writeProtocolHeader(dOut)
      writeByteBufferToStream(metaBuf, dOut)
      writeHostBufferToStream(dataBuf, dOut)
      headerLen + metaBuf.capacity() + dataBuf.getLength
    }
  }
}

private[rapids] class SimpleTableDeserializer(
    sparkTypes: Array[DataType],
    deserTime: GpuMetric) extends TableSerde {
  private def readProtocolHeader(dIn: DataInputStream): Unit = {
    val magicNum = dIn.readInt()
    if (magicNum != P_MAGIC_NUM) {
      throw new IllegalStateException(s"Expected magic number $P_MAGIC_NUM for " +
        s"table serializer, but got $magicNum")
    }
    val version = dIn.readInt()
    if (version != P_VERSION) {
      throw new IllegalStateException(s"Version mismatch: expected $P_VERSION for " +
        s"table serializer, but got $version")
    }
  }

  private def readByteBufferFromStream(dIn: DataInputStream): ByteBuffer = {
    val bufLen = dIn.readLong().toInt
    val bufArray = new Array[Byte](bufLen)
    var readLen = 0
    // A single call to read(bufArray) can not always read the expected length. So
    // we do it here ourselves.
    do {
      val ret = dIn.read(bufArray, readLen, bufLen - readLen)
      if (ret < 0) {
        throw new EOFException()
      }
      readLen += ret
    } while (readLen < bufLen)
    ByteBuffer.wrap(bufArray)
  }

  private def readHostBufferFromStream(dIn: DataInputStream): HostMemoryBuffer = {
    val bufLen = dIn.readLong()
    closeOnExcept(getHostBuffer(bufLen)) { hostBuf =>
      var leftLen = bufLen
      var hOffset = 0L
      while (leftLen > 0) {
        val copyLen = Math.min(tmpBuf.length, leftLen)
        val readLen = dIn.read(tmpBuf, 0, copyLen.toInt)
        if (readLen < 0) {
          throw new EOFException()
        }
        hostBuf.setBytes(hOffset, tmpBuf, 0, readLen)
        hOffset += readLen
        leftLen -= readLen
      }
      hostBuf
    }
  }

  def readFromStream(dIn: DataInputStream): ColumnarBatch = {
    // IO operation is coming, so leave GPU for a while.
    GpuSemaphore.releaseIfNecessary(TaskContext.get())
    val tableMeta = deserTime.ns {
      // 1) read and check header
      readProtocolHeader(dIn)
      // 2) read table metadata
      TableMeta.getRootAsTableMeta(readByteBufferFromStream(dIn))
    }
    if (tableMeta.packedMetaAsByteBuffer() == null) {
      // no packed metadata, must be a table with zero columns
      // Acquiring the GPU even the coming batch is empty, because the downstream
      // tasks expect the GPU batch producer to acquire the semaphore and may
      // generate GPU data from batches that are empty.
      GpuSemaphore.acquireIfNecessary(TaskContext.get())
      new ColumnarBatch(Array.empty, tableMeta.rowCount().toInt)
    } else {
      // 3) read table data
      val hostBuf = withResource(new NvtxRange("Read Host Table", NvtxColor.ORANGE)) { _ =>
        deserTime.ns(readHostBufferFromStream(dIn))
      }
      val data = withResource(hostBuf) { _ =>
        // Begin to use GPU
        GpuSemaphore.acquireIfNecessary(TaskContext.get())
        withResource(new NvtxRange("Table to Device", NvtxColor.YELLOW)) { _ =>
          deserTime.ns {
            closeOnExcept(DeviceMemoryBuffer.allocate(hostBuf.getLength)) { devBuf =>
              devBuf.copyFromHostBuffer(hostBuf)
              devBuf
            }
          }
        }
      }
      withResource(new NvtxRange("Deserialize Table", NvtxColor.RED)) { _ =>
        deserTime.ns {
          withResource(data) { _ =>
            val bufferMeta = tableMeta.bufferMeta()
            if (bufferMeta == null || bufferMeta.codecBufferDescrsLength == 0) {
              MetaUtils.getBatchFromMeta(data, tableMeta, sparkTypes)
            } else {
              GpuCompressedColumnVector.from(data, tableMeta)
            }
          }
        }
      }
    }
  }

}

private[rapids] class SerializedTableIterator(dIn: DataInputStream,
    sparkTypes: Array[DataType],
    deserTime: GpuMetric) extends Iterator[(Int, ColumnarBatch)] {

  private val tableDeserializer = new SimpleTableDeserializer(sparkTypes, deserTime)
  private var closed = false
  private var onDeck: Option[SpillableColumnarBatch] = None
  Option(TaskContext.get()).foreach { tc =>
    onTaskCompletion(tc) {
      onDeck.foreach(_.close())
      onDeck = None
      tableDeserializer.close()
      if (!closed) {
        dIn.close()
      }
    }
  }

  override def hasNext: Boolean = {
    if (onDeck.isEmpty) {
      tryReadNextBatch()
    }
    onDeck.isDefined
  }

  override def next(): (Int, ColumnarBatch) = {
    if (!hasNext) {
      throw new NoSuchElementException()
    }
    val ret = withResource(onDeck) { _ =>
      onDeck.get.getColumnarBatch()
    }
    onDeck = None
    (0, ret)
  }

  private def tryReadNextBatch(): Unit = {
    if (closed) {
      return
    }
    try {
      onDeck = Some(SpillableColumnarBatch(tableDeserializer.readFromStream(dIn),
        SpillPriorities.ACTIVE_ON_DECK_PRIORITY))
    } catch {
      case _: EOFException => // we reach the end
        dIn.close()
        closed = true
        onDeck.foreach(_.close())
        onDeck = None
    }
  }
}
