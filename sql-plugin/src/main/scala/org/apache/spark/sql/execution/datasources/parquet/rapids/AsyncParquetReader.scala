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
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.atomic.{AtomicBoolean, AtomicInteger, AtomicLong}
import java.util.concurrent.locks.ReentrantLock

import scala.collection.JavaConverters._
import scala.collection.mutable

import ai.rapids.cudf.{HostColumnVector, HostMemoryBuffer, NvtxColor, Table}
import com.nvidia.spark.rapids.{DateTimeRebaseMode, GpuDataProducer, GpuMetric, GpuSemaphore, HMBInputFile, NvtxWithMetrics}
import com.nvidia.spark.rapids.Arm.{closeOnExcept, withResource}
import com.nvidia.spark.rapids.RapidsPluginImplicits._
import org.apache.hadoop.conf.Configuration
import org.apache.parquet.{HadoopReadOptions, VersionParser}
import org.apache.parquet.VersionParser.ParsedVersion
import org.apache.parquet.column.page.PageReadStore
import org.apache.parquet.hadoop.ParquetFileReader
import org.apache.parquet.schema.MessageType
import org.json4s._
import org.json4s.DefaultFormats
import org.json4s.jackson.JsonMethods._

import org.apache.spark.TaskContext
import org.apache.spark.internal.Logging
import org.apache.spark.sql.execution.datasources.parquet.rapids.HybridTableProducer.preloadMemPool
import org.apache.spark.sql.execution.vectorized.rapids.HostWritableColumnVector
import org.apache.spark.sql.rapids.ComputeThreadPool
import org.apache.spark.sql.rapids.ComputeThreadPool.TaskWithPriority
import org.apache.spark.sql.types.{ArrayType, BinaryType, DataType, DecimalType, MapType, StructType}


case class AsyncBatchResult(data: Array[HostColumnVector], sizeInByte: Long)

class AsyncParquetReader(
    conf: Configuration,
    tgtBatchSize: Int,
    fileBuffer: HostMemoryBuffer,
    offset: Long,
    len: Long,
    metrics: Map[String, GpuMetric],
    dateRebaseMode: DateTimeRebaseMode,
    timestampRebaseMode: DateTimeRebaseMode,
    clippedSchema: MessageType)
  extends Iterator[AsyncBatchResult] with AutoCloseable with Logging {

  private val initialized = new AtomicBoolean(false)
  @volatile private var cancelled = false
  private val taskID = TaskContext.get().taskAttemptId()
  private val runningLock = new ReentrantLock()

  private lazy val resultQueue = new LinkedBlockingQueue[AsyncBatchResult]()

  private var task: TaskWithPriority[Unit] = _

  private var currentGroup: PageReadStore = _

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

  private lazy val rowGroupQueue: mutable.Queue[PageReadStore] = {
    val rowGroups = mutable.Queue.empty[PageReadStore]
    var curRowGroup: PageReadStore = pageReader.readNextRowGroup()
    while (curRowGroup != null) {
      rowGroups += curRowGroup;
      curRowGroup = pageReader.readNextRowGroup()
    }
    rowGroups
  }

  private lazy val totalRowCnt = {
    remainTotalRows = rowGroupQueue.foldLeft(0)((s, x) => s + x.getRowCount.toInt)
    remainTotalRows
  }
  private lazy val rowBatchSize = {
    val value = (tgtBatchSize.toDouble / len * totalRowCnt).toInt max 1
    remainBatchRows = value min totalRowCnt
    remainBatchRows
  }
  private var remainTotalRows: Int = _
  private var remainBatchRows: Int = _
  private var remainPageRows: Int = 0

  private lazy val numBatches = {
    val value = (totalRowCnt + rowBatchSize - 1) / rowBatchSize
    logError(s"[$taskID] rowCount:$totalRowCnt; rowBatchSize:$rowBatchSize; numBatches:$value")
    value
  }
  private val consumedBatches = new AtomicInteger(0)
  private val producedBatches = new AtomicInteger(0)

  private lazy val hostColumnBuilders: Array[HostWritableColumnVector] = {
    parquetColumn.sparkType.asInstanceOf[StructType].fields.map { f =>
      new HostWritableColumnVector(rowBatchSize min totalRowCnt, f.dataType)
    }
  }

  private lazy val columnVectors: Array[ParquetColumnVector] =  {
    initialized.getAndSet(true)

    hostColumnBuilders.indices.toArray.map { i =>
      new ParquetColumnVector(parquetColumn.children(i),
        hostColumnBuilders(i),
        rowBatchSize min totalRowCnt,
        Set.empty[ParquetColumn].asJava, true, -1, null);
    }
  }

  private def releaseParquetCV(parquetCVs: Array[ParquetColumnVector]): Unit = {
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
        releaseParquetCV(pcv.getChildren.asScala.toArray)
      }
    }
  }

  private def readImpl(): AsyncBatchResult = metrics("cpuDecodeTime").ns {
    while (remainBatchRows > 0) {
      if (currentGroup == null || remainPageRows == 0) {
        currentGroup = rowGroupQueue.dequeue()
        remainPageRows = currentGroup.getRowCount.toInt
        // update column readers to read the new page
        val stack = mutable.Stack[ParquetColumnVector](columnVectors: _*)
        while (stack.nonEmpty) {
          stack.pop() match {
            case cv if cv.getColumn.isPrimitive =>
              cv.setColumnReader(
                new VectorizedColumnReader(
                  cv.getColumn.descriptor.get,
                  cv.getColumn.required,
                  cv.maxRepetitiveDefLevel,
                  currentGroup,
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

      val readSize = remainBatchRows min remainPageRows
      remainPageRows -= readSize
      remainBatchRows -= readSize
      remainTotalRows -= readSize

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
    // materialize current batch in the memory layout of cuDF column vector
    val sizeCounter: Array[Long] = Array.fill(1)(0L)
    val result = hostColumnBuilders.map(_.build(sizeCounter))
    // update batch size and remaining
    remainBatchRows = rowBatchSize min remainTotalRows
    // Reset all the HostColumnBuffers for the upcoming batch
    hostColumnBuilders.foreach(_.reallocate(remainBatchRows))

    AsyncBatchResult(result, sizeCounter.head)
  }

  private def readAsProducer(): Unit = {
    // Ensure all lazy variables are initialized before the start of decoding
    columnVectors

    var numProduced = producedBatches.get()
    while (numProduced < numBatches && !cancelled) {
      val batchResult = try {
        runningLock.lock()
        if (cancelled) None else Option(readImpl())
      } finally {
        runningLock.unlock()
      }
      batchResult.foreach { ret =>
        resultQueue.offer(ret)
        numProduced = producedBatches.incrementAndGet()
        logDebug(s"[$taskID] produced a new batch($numProduced/$numBatches)")
      }
    }
  }

  private def launchTask(): Unit = if (task == null) {
    task = new TaskWithPriority(() => readAsProducer(), 0, true)
    ComputeThreadPool.submitTask(task)
  }

  override def hasNext: Boolean = {
    launchTask()
    consumedBatches.get() < numBatches
  }

  override def next(): AsyncBatchResult = {
    val ret = resultQueue.take()
    val numConsumed = consumedBatches.incrementAndGet()
    logDebug(s"[$taskID] consumed a new batch($numConsumed/$numBatches)")
    ret
  }

  override def close(): Unit = {
    if (task != null && !cancelled) {
      if (producedBatches.get() < numBatches) {
        logError(s"[$taskID] Close while producer still running asynchronously(total:$numBatches" +
          s"/consumed:${consumedBatches.get()}/produced:${producedBatches.get()})")
      }
      task.cancel()
      cancelled = true
    }

    runningLock.lock()
    // release all prefetched batches
    while (!resultQueue.isEmpty) {
      resultQueue.take().data.foreach(_.close())
    }
    // release all work buffers since all work are done
    if (initialized.get()) {
      releaseParquetCV(columnVectors)
    }
    // close ParquetFileReader to release all decompressors
    pageReader.close()
    // close fileBuffer additionally to reduce refCount to 0
    if (fileBuffer.getRefCount > 0) {
      fileBuffer.close()
    }
    runningLock.unlock()
  }

}

object AsyncParquetReader {
  def apply(conf: Configuration,
            tgtBatchSize: Int,
            fileBuffer: HostMemoryBuffer,
            offset: Long,
            len: Long,
            metrics: Map[String, GpuMetric],
            dateRebaseMode: DateTimeRebaseMode,
            timestampRebaseMode: DateTimeRebaseMode,
            hasInt96Timestamps: Boolean,
            clippedSchema: MessageType,
            readDataSchema: StructType): AsyncParquetReader = {
    new AsyncParquetReader(conf,
      tgtBatchSize, fileBuffer, offset, len,
      metrics,
      dateRebaseMode, timestampRebaseMode,
      clippedSchema)
  }
}

class HybridTableProducer(
    asyncHostReader: AsyncParquetReader,
    hybridOpts: HybridParquetOpts,
    metrics: Map[String, GpuMetric]) extends GpuDataProducer[Table] with Logging {

  private val enablePreload = hybridOpts.maxDevicePreloadBytes > 0
  private var preloadedMemSize: Long = 0
  private var holdGPUSemaphore = false

  override def hasNext: Boolean = {
    asyncHostReader.hasNext
  }

  override def next: Table = {
    val asyncRet = withResource(new NvtxWithMetrics("waitAsyncCpuDecode", NvtxColor.YELLOW,
      metrics("waitAsyncDecode"))) { _ =>
      asyncHostReader.next()
    }

    if (!holdGPUSemaphore) {
      if (enablePreload && tryAcquirePreloadH2DSlots(asyncRet.sizeInByte)) {
        preloadedMemSize += asyncRet.sizeInByte
        metrics("preloadH2DBatches") += 1
      } else {
        withResource(new NvtxWithMetrics(
          "GPU wait time before H2D", NvtxColor.WHITE, metrics("preH2dGpuWait"))) { _ =>
          takeSemaphore()
        }
      }
    }

    withResource(new NvtxWithMetrics("Transfer HostVectors to Device", NvtxColor.CYAN,
      metrics("hostVecToDeviceTime"))) { _ =>
      closeOnExcept(asyncRet.data) { hostCVs =>

        val batchRows = hostCVs.head.getRowCount
        logInfo(s"VectorizedParquetGpuProducer batches $batchRows rows; ")
        metrics.get("cpuDecodeRows").foreach(_.+=(batchRows))
        metrics.get("cpuDecodeBatches").foreach(_.+=(1))
        metrics.get("numOutputBatches").foreach(_.+=(1))

        val deviceCVs = hostCVs.safeMap { hcv =>
          val dcv = hcv.copyToDevice()
          hcv.close()
          dcv
        }
        withResource(deviceCVs) { dCVs => new Table(dCVs: _*) }
      }
    }
  }

  private def tryAcquirePreloadH2DSlots(batchMemSize: Long): Boolean = {
    if (preloadMemPool.addAndGet(-batchMemSize) < 0) {
      preloadMemPool.getAndAdd(batchMemSize)
      false
    } else {
      true
    }
  }

  private def takeSemaphore(): Unit = {
    GpuSemaphore.acquireIfNecessary(TaskContext.get())
    holdGPUSemaphore = true
    if (enablePreload) {
      preloadMemPool.getAndAdd(preloadedMemSize)
      preloadedMemSize = 0
    }
  }

  override def close(): Unit = {
    asyncHostReader.close()
    if (!holdGPUSemaphore) {
      withResource(new NvtxWithMetrics(
        "GPU wait time after H2D", NvtxColor.WHITE, metrics("postH2dGpuWait"))) { _ =>
        takeSemaphore()
      }
    }
  }

}

object HybridTableProducer {
  def parseHybridParquetOpts(str: String): HybridParquetOpts = {
    implicit val formats: Formats = DefaultFormats
    val defaultJson = parse(
      """{"mode": "GPU_ONLY", "maxHostThreads": 0, "batchSizeBytes": 0,
        |"pollInterval": 0, "maxDevicePreloadBytes": 0}""".stripMargin)

    val json: JValue = if (str.isEmpty) {
      defaultJson
    } else {
      defaultJson merge parse(str)
    }
    json.extract[HybridParquetOpts]
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

  def initialize(opts: HybridParquetOpts): Unit = {
    if (!initialized.get()) synchronized {
      if (!initialized.get()) {
        ComputeThreadPool.launch(opts.maxHostThreads, 1024)
        preloadMemPool = new AtomicLong(opts.maxDevicePreloadBytes)
        initialized.set(true)
      }
    }
  }

  private var preloadMemPool: AtomicLong = _
  private val initialized: AtomicBoolean = new AtomicBoolean(false)

}

// sealed trait ReadMode
// object HostOnly extends ReadMode
// object DeviceOnly extends ReadMode
// object DeviceFirst extends ReadMode

case class HybridParquetOpts(mode: String,
                             maxHostThreads: Int,
                             batchSizeBytes: Long,
                             pollInterval: Int,
                             maxDevicePreloadBytes: Long)
