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

package org.apache.spark.sql.execution.datasources.parquet.rapids

import java.util.TimeZone
import java.util.concurrent.atomic.AtomicInteger

import scala.collection.JavaConverters._
import scala.collection.mutable

import ai.rapids.cudf.{HostColumnVector, HostMemoryBuffer, NvtxColor, Table}
import com.nvidia.spark.rapids._
import com.nvidia.spark.rapids.Arm.withResource
import org.apache.hadoop.conf.Configuration
import org.apache.parquet.{HadoopReadOptions, VersionParser}
import org.apache.parquet.VersionParser.ParsedVersion
import org.apache.parquet.column.page.PageReadStore
import org.apache.parquet.hadoop.ParquetFileReader
import org.apache.parquet.schema.MessageType

import org.apache.spark.TaskContext
import org.apache.spark.internal.Logging
import org.apache.spark.sql.execution.vectorized.rapids.HostWritableColumnVector
import org.apache.spark.sql.types.{ArrayType, BinaryType, DataType, DecimalType, MapType, StructType}


class HostParquetProducer(
    conf: Configuration,
    tgtBatchSize: Int,
    fileBuffer: HostMemoryBuffer,
    offset: Long,
    len: Long,
    metrics: Map[String, GpuMetric],
    dateRebaseMode: DateTimeRebaseMode,
    timestampRebaseMode: DateTimeRebaseMode,
    hasInt96Timestamps: Boolean,
    clippedSchema: MessageType,
    readDataSchema: StructType) extends GpuDataProducer[Table] with Logging {

  logDebug(s"ColumnDescriptors ${clippedSchema.getColumns.asScala.mkString(" | ")}")
  logDebug(s"ColumnFieldTypes ${clippedSchema.asGroupType().getFields.asScala.mkString(" | ")}")
  logDebug(s"ReadDataSchema ${readDataSchema.sql}")

  private var readerClosed = false

  private val pageReader: ParquetFileReader = {
    val options = HadoopReadOptions.builder(conf)
        .withRange(offset, offset + len)
        .withCodecFactory(new ParquetCodecFactory(conf, 0))
        .build()
    val bufferFile = new HMBInputFile(fileBuffer, length = Some(offset + len))
    val reader = new ParquetFileReader(bufferFile, options)
    // The fileSchema here has already been clipped
    reader.setRequestedSchema(clippedSchema)
    reader
  }

  private val parquetColumn: ParquetColumn = {
    val converter = new ParquetToSparkSchemaConverter()
    converter.convertParquetColumn(clippedSchema, None)
  }

  private val writerVersion: ParsedVersion = try {
    VersionParser.parse(pageReader.getFileMetaData.getCreatedBy)
  } catch {
    case _: Exception =>
      // If any problems occur trying to parse the writer version, fallback to sequential reads
      // if the column is a delta byte array encoding (due to PARQUET-246).
      null
  }

  private var hostColumnBuilders: Array[HostWritableColumnVector] = _

  private var columnVectors: Array[ParquetColumnVector] = _

  private def createEverything(batchSize: Int): Unit = {
    hostColumnBuilders = parquetColumn.sparkType.asInstanceOf[StructType].fields.map { f =>
      new HostWritableColumnVector(batchSize, f.dataType)
    }
    columnVectors = hostColumnBuilders.indices.toArray.map { i =>
      new ParquetColumnVector(parquetColumn.children(i),
        hostColumnBuilders(i), batchSize,
        Set.empty[ParquetColumn].asJava, true, -1, null);
    }
  }

  private def releaseEverything(parquetCVs: Array[ParquetColumnVector]): Unit = {
    parquetCVs.foreach { pcv =>
      pcv.getValueVector.close()
      if (pcv.getColumn.isPrimitive) {
        pcv.setColumnReader(null)
        if (pcv.getDefinitionLevelVector != null) {
          pcv.getDefinitionLevelVector.close()
        }
        if (pcv.getRepetitionLevelVector != null) {
          pcv.getRepetitionLevelVector.close()
        }
      }
      if (pcv.getChildren.size() > 0) {
        releaseEverything(pcv.getChildren.asScala.toArray)
      }
    }
  }

  // Performed all the host-side reading work before transferring to device
  private lazy val hostBatches: mutable.Queue[Array[HostColumnVector]] = {

    val buffer = mutable.Queue.empty[Array[HostColumnVector]]

    val rowGroupBuilder = mutable.ArrayBuffer.empty[PageReadStore]
    do {
      rowGroupBuilder += pageReader.readNextRowGroup()
    } while (rowGroupBuilder.last != null)
    val rowGroups = rowGroupBuilder.dropRight(1).toArray

    val totalRowCnt = rowGroups.foldLeft(0)((s, x) => s + x.getRowCount.toInt)
    val rowBatchSize = (tgtBatchSize.toDouble / len * totalRowCnt).toInt max 1
    logInfo(s"total row count: $totalRowCnt ; batch size in row: $rowBatchSize")

    var remainTotalRows = totalRowCnt
    var remainBatchRows = rowBatchSize min totalRowCnt
    createEverything(remainBatchRows)

    rowGroups.foreach { rowGroup: PageReadStore =>
      // update column readers to read the new page
      metrics("cpuDecodeDictTime").ns {
        val stack = mutable.Stack[ParquetColumnVector](columnVectors: _*)
        while (stack.nonEmpty) {
          stack.pop() match {
            case cv if cv.getColumn.isPrimitive =>
              cv.setColumnReader(
                new VectorizedColumnReader(
                  cv.getColumn.descriptor.get,
                  cv.getColumn.required,
                  cv.maxRepetitiveDefLevel,
                  rowGroup,
                  null,
                  dateRebaseMode.value,
                  TimeZone.getDefault.getID,
                  timestampRebaseMode.value,
                  TimeZone.getDefault.getID,
                  writerVersion))
            case cv =>
              cv.getChildren.asScala.foreach(stack.push)
          }
        }
      }

      var remainPageRows = rowGroup.getRowCount.toInt
      while (remainPageRows > 0) {
        val readSize = remainBatchRows min remainPageRows
        remainPageRows -= readSize
        remainBatchRows -= readSize
        remainTotalRows -= readSize

        metrics("cpuDecodeDataTime").ns {
          columnVectors.foreach { cv =>
            cv.getLeaves.asScala.foreach {
              case leaf if leaf.getColumnReader != null =>
                leaf.getColumnReader.readBatch(readSize, leaf.getValueVector,
                  leaf.getRepetitionLevelVector, leaf.getDefinitionLevelVector)
              case _ =>
            }
            cv.assemble()
            // Reset all value vectors(HostWritableColumnVector) along with def/repVectors.
            // The reset is essential because we are going to either finalize current batch
            // or read another RowGroup, or even both.
            // As of value vectors, reset means update the offsets of target buffers.
            // As of def/repVectors vectors, reset simply means re-initialize.
            cv.reset()
          }
        }

        // Finalize current batch and reset all buffers for the next batch
        if (remainBatchRows == 0) {
          metrics("hostVecBuildTime").ns {
            // materialize current batch in the memory layout of cuDF column vector
            buffer.enqueue(hostColumnBuilders.map(_.build()))
            // update batch size and remaining
            remainBatchRows = rowBatchSize min remainTotalRows
            // Reset all the HostColumnBuffers for the upcoming batch
            hostColumnBuilders.foreach(_.reallocate(remainBatchRows))
          }
        }
      }
    }

    // release all work buffers since all work are done
    releaseEverything(columnVectors)

    // close file buffer ASAP
    fileBuffer.close()
    // close ParquetFileReader to release all decompressors
    pageReader.close()
    readerClosed = true

    // release host resource slot if allocated
    HostParquetProducer.hostResourceSemaphore.foreach(_.getAndIncrement())

    buffer
  }

  private var firstBatch = true

  override def hasNext: Boolean = {
    if (firstBatch) {
      try {
        withResource(new NvtxWithMetrics("cpuDecodeTime", NvtxColor.GREEN,
          metrics("cpuDecodeTime"))) { _ =>
          hostBatches
        }
      } catch {
        case _: Throwable =>
          hostBatches.foreach(batch => batch.foreach(_.close()))
      }
    }
    hostBatches.nonEmpty
  }

  override def next: Table = {
    withResource(hostBatches.dequeue()) { hostCVs =>
      if (firstBatch) {
        // About to start using the GPU
        GpuSemaphore.acquireIfNecessary(TaskContext.get())
        firstBatch = false
      }

      val batchRows = hostCVs.head.getRowCount
//      logInfo(s"VectorizedParquetGpuProducer batches $batchRows rows; " +
//        s"PinnedPoolSize:${PinnedMemoryPool.getTotalPoolSizeBytes};" +
//        s" remain:${PinnedMemoryPool.getAvailableBytes}")
      metrics.get("cpuDecodeRows").foreach(_.+=(batchRows))
      metrics.get("cpuDecodeBatches").foreach(_.+=(1))
      metrics.get("numOutputBatches").foreach(_.+=(1))

      withResource(new NvtxWithMetrics("Transfer HostVectors to Device", NvtxColor.CYAN,
        metrics.get("hostVecToDeviceTime").toArray: _*)) { _ =>
        withResource(hostCVs.indices.map(i => hostCVs(i).copyToDevice())) { dCVs =>
          new Table(dCVs: _*)
        }
      }
    }
  }

  override def close(): Unit = {
    if (!firstBatch) {
      hostBatches.foreach { hcvArray => hcvArray.foreach(_.close()) }
    }
    if(!readerClosed) {
      pageReader.close()
      fileBuffer.close()
    }
  }
}

object HostParquetProducer {

  def parseHybridParquetOpts(str: String): HybridParquetOpts = {
    str match {
      case "" =>
        HybridParquetOpts(DeviceOnly, 0, 0)
      case "GPU_ONLY" =>
        HybridParquetOpts(DeviceOnly, 0, 0)
      case "CPU_ONLY" =>
        HybridParquetOpts(HostOnly, 0, 0)
      case s if s.startsWith("DeviceFirst(") && s.endsWith(")") =>
        val args = s.slice(12, str.length -1).split(",").map(_.toInt)
        HybridParquetOpts(DeviceFirst, args(0), args(1))
      case _ =>
        throw new IllegalArgumentException(s"illegal HybridParquetOpts $str")
    }
  }

  def schemaSupportCheck(types: Array[DataType]): Boolean = {
    types.collectFirst {
      case _: BinaryType =>
        false
      case _: DecimalType =>
        // if DecimalType.isByteArrayDecimalType(dt) =>
        false
      case st: StructType =>
        schemaSupportCheck(st.fields.map(_.dataType))
      case ArrayType(et, _) =>
        schemaSupportCheck(Array(et))
      case MapType(kt, vt, _) =>
        schemaSupportCheck(Array(kt, vt))
    }.getOrElse(true)
  }

  private var hostResourceSemaphore: Option[AtomicInteger] = None

  def acquireSlot(totalSize: Int): Boolean = {
    if (hostResourceSemaphore.isEmpty) {
      synchronized {
        if (hostResourceSemaphore.isEmpty) {
          hostResourceSemaphore = Some(new AtomicInteger(totalSize))
        }
      }
    }
    if (hostResourceSemaphore.get.getAndDecrement() < 1) {
      hostResourceSemaphore.get.getAndIncrement()
      false
    } else {
      true
    }
  }

}

sealed trait ReadMode
object HostOnly extends ReadMode
object DeviceOnly extends ReadMode
object DeviceFirst extends ReadMode

case class HybridParquetOpts(mode: ReadMode, hostCurrentBound: Int, pollInterval: Int)
