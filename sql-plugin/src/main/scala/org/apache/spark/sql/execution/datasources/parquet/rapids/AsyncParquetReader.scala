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
import java.util.concurrent.{LinkedBlockingQueue, Semaphore, TimeUnit}
import java.util.concurrent.atomic.{AtomicBoolean, AtomicInteger, AtomicLong}
import java.util.concurrent.locks.ReentrantLock

import scala.collection.JavaConverters._
import scala.collection.mutable

import ai.rapids.cudf.{HostColumnVector, HostMemoryBuffer, NvtxColor, Table}
import com.nvidia.spark.rapids.{DateTimeRebaseMode, GpuColumnVector, GpuMetric, GpuSemaphore, HMBInputFile, NvtxWithMetrics}
import com.nvidia.spark.rapids.Arm.withResource
import com.nvidia.spark.rapids.RapidsPluginImplicits._
import com.nvidia.spark.rapids.RmmRapidsRetryIterator.{AutoCloseableAttemptSpliterator, RmmRapidsRetryAutoCloseableIterator}
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
import org.apache.spark.sql.execution.vectorized.rapids.HostWritableColumnVector
import org.apache.spark.sql.rapids.ComputeThreadPool
import org.apache.spark.sql.rapids.ComputeThreadPool.TaskWithPriority
import org.apache.spark.sql.types.{ArrayType, BinaryType, DataType, DecimalType, MapType, StructType}
import org.apache.spark.sql.vectorized.ColumnarBatch


case class AsyncBatchResult(data: Array[HostColumnVector],
                            sizeInByte: Long) extends AutoCloseable {
  override def close(): Unit = {
    data.safeClose()
  }
}

class AsyncParquetReaderError(ex: Throwable) extends RuntimeException(ex)

class AsyncParquetReader(
    conf: Configuration,
    tgtBatchSize: Int,
    fileBuffer: HostMemoryBuffer,
    offset: Long,
    len: Long,
    metrics: Map[String, GpuMetric],
    dateRebaseMode: DateTimeRebaseMode,
    timestampRebaseMode: DateTimeRebaseMode,
    clippedSchema: MessageType,
    slotAcquired: Boolean,
    asynchronous: Boolean = true)
  extends Iterator[AsyncBatchResult] with AutoCloseable with Logging {

  private type Element = Either[AsyncBatchResult, Throwable]

  private val initialized = new AtomicBoolean(false)
  @volatile private var cancelled = false
  private val taskID = TaskContext.get().taskAttemptId()

  private val runningLock = if (asynchronous) {
    Some(new ReentrantLock())
  } else {
    None
  }

  private lazy val resultQueue = new LinkedBlockingQueue[Element]()

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
      val batchResult: Element = try {
        runningLock.get.lock()
        if (cancelled) {
          Right(new RuntimeException("AsyncReadTask has been cancelled"))
        } else {
          Left(readImpl())
        }
      } catch {
        case ex: Throwable =>
          Right(ex)
      } finally {
        runningLock.get.unlock()
      }

      if (!cancelled) {
        resultQueue.offer(batchResult)
        batchResult match {
          case Left(_) =>
            numProduced = producedBatches.incrementAndGet()
            logError(s"[$taskID] produced a new batch($numProduced/$numBatches)")
          case Right(exception) =>
            throw exception
        }
      }
    }
  }

  private def launchTask(): Unit = if (asynchronous && task == null) {
    //    val failedCallback = new FailedCallback {
    //      private val queue = resultQueue
    //      override def callback(ex: Throwable): Unit = {
    //        queue.offer(Right(ex))
    //      }
    //    }
    task = new TaskWithPriority(() => readAsProducer(),
      0, slotAcquired, null, null)
    ComputeThreadPool.submitTask(task)
  }

  override def hasNext: Boolean = {
    launchTask()
    consumedBatches.get() < numBatches
  }

  override def next(): AsyncBatchResult = {
    withResource(new NvtxWithMetrics("waitAsyncCpuDecode", NvtxColor.YELLOW,
      metrics("waitAsyncDecode"))) { _ =>

      val batchResult = if (asynchronous) {
        resultQueue.take()
      } else {
        Left(readImpl())
      }
      batchResult match {
        case Left(columnBatch) =>
          val numConsumed = consumedBatches.incrementAndGet()
          logError(s"[$taskID] consumed a new batch($numConsumed/$numBatches)")
          columnBatch
        case Right(ex) =>
          throw new AsyncParquetReaderError(ex)
      }
    }
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

    runningLock.foreach(_.lock())
    // release all prefetched batches
    while (asynchronous && !resultQueue.isEmpty) {
      resultQueue.take() match {
        case Left(data) => data.safeClose()
        case Right(_) =>
      }
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
    runningLock.foreach(_.unlock())

    // release the slot for synchronous mode
    if (!asynchronous) {
      HostParquetIterator.releaseSlot(false)
    }
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
            readDataSchema: StructType,
            slotAcquired: Boolean,
            asynchronous: Boolean): AsyncParquetReader = {
    new AsyncParquetReader(conf,
      tgtBatchSize, fileBuffer, offset, len,
      metrics,
      dateRebaseMode, timestampRebaseMode,
      clippedSchema,
      slotAcquired, asynchronous)
  }
}

class HostParquetIterator(
    asyncIter: HostParquetIterator.HostParquetSpliterator) extends
  RmmRapidsRetryAutoCloseableIterator[AsyncBatchResult, ColumnarBatch](asyncIter) {

  private var isClosed = false

  override def hasNext: Boolean = {
    val hasNext = !isClosed && super.hasNext
    if (!hasNext && !isClosed) {
      asyncIter.close()
      isClosed = true
    }
    hasNext
  }
}

object HostParquetIterator extends Logging {

  def apply(asyncReader: AsyncParquetReader,
            hybridOpts: HybridParquetOpts,
            dataTypes: Array[DataType],
            metrics: Map[String, GpuMetric]): HostParquetIterator = {
    val h2dTransfer = new HostToDeviceTransfer(dataTypes, hybridOpts, metrics)
    val spliterator = new HostParquetSpliterator(asyncReader, h2dTransfer)
    new HostParquetIterator(spliterator)
  }

  private class HostParquetSpliterator(
      asyncReader: AsyncParquetReader,
      transfer: HostToDeviceTransfer) extends
    AutoCloseableAttemptSpliterator[AsyncBatchResult, ColumnarBatch](
      asyncReader, transfer.execute) {

    override def close(): Unit = {
      asyncReader.close()
      transfer.close()
      super.close()
    }
  }

  private class HostToDeviceTransfer(
      dataTypes: Array[DataType],
      hybridOpts: HybridParquetOpts,
      metrics: Map[String, GpuMetric]) extends AutoCloseable with Logging {

    private val enablePreload = hybridOpts.maxPreloadBytes > 0

    def execute(batchResult: AsyncBatchResult): ColumnarBatch = {
      val batchMemSize = batchResult.sizeInByte

      val preloaded = if (!enablePreload || !tryAcquirePreloadH2DSlots(batchMemSize)) {
        withResource(new NvtxWithMetrics(
          "GPU wait time before H2D", NvtxColor.WHITE, metrics("preH2dGpuWait"))) { _ =>
          GpuSemaphore.acquireIfNecessary(TaskContext.get())
        }
        false
      } else {
        metrics("preloadH2DBatches") += 1
        true
      }

      val ret = withResource(new NvtxWithMetrics("Transfer HostVectors to Device", NvtxColor.CYAN,
        metrics("hostVecToDeviceTime"))) { _ =>

        // Do NOT close hostColumnVectors here in case of retry (it will be closed via
        // AsyncBatchResult::close)
        val deviceCVs = batchResult.data.safeMap(_.copyToDevice())

        val batchRows = batchResult.data.head.getRowCount
        metrics.get("cpuDecodeRows").foreach(_.+=(batchRows))
        metrics.get("cpuDecodeBatches").foreach(_.+=(1))
        metrics.get("numOutputBatches").foreach(_.+=(1))

        withResource(deviceCVs) { dCVs =>
          withResource(new Table(dCVs: _*)) { table =>
            GpuColumnVector.from(table, dataTypes)
          }
        }
      }

      if (preloaded) {
        withResource(new NvtxWithMetrics(
          "GPU wait time after H2D", NvtxColor.WHITE, metrics("postH2dGpuWait"))) { _ =>
          GpuSemaphore.acquireIfNecessary(TaskContext.get())
          preloadMemPool.getAndAdd(batchMemSize)
        }
      }

      ret
    }

    private def tryAcquirePreloadH2DSlots(batchMemSize: Long): Boolean = {
      if (preloadMemPool.addAndGet(-batchMemSize) < 0) {
        preloadMemPool.getAndAdd(batchMemSize)
        false
      } else {
        true
      }
    }

    override def close(): Unit = {}
  }

  def parseHybridParquetOpts(str: String): HybridParquetOpts = {
    implicit val formats: Formats = DefaultFormats
    val defaultJson = parse(
      """{"mode": "GPU_ONLY", "maxConcurrent": 0, "batchSizeBytes": 0, "async": true,
        |"pollInterval": 0, "maxPreloadBytes": 0}""".stripMargin)

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

  def tryAcquireSlot(opts: HybridParquetOpts): Boolean = {
    if (opts.async) {
      ComputeThreadPool.bookIdleWorker(opts.pollInterval)
    } else {
      syncSemaphore.tryAcquire(1, opts.pollInterval, TimeUnit.MILLISECONDS)
    }
  }

  def releaseSlot(async: Boolean): Unit = if (async) {
    ComputeThreadPool.releaseWorker()
  } else {
    syncSemaphore.release()
  }

  def initialize(opts: HybridParquetOpts): Unit = {
    if (!initialized.get()) synchronized {
      if (!initialized.get()) {
        syncSemaphore = new Semaphore(opts.maxConcurrent)
        ComputeThreadPool.launch(opts.maxConcurrent, 1024)
        preloadMemPool = new AtomicLong(opts.maxPreloadBytes)
        initialized.set(true)
      }
    }
  }

  private var preloadMemPool: AtomicLong = _
  private var syncSemaphore: Semaphore = _
  private val initialized: AtomicBoolean = new AtomicBoolean(false)
}

// sealed trait ReadMode
// object HostOnly extends ReadMode
// object DeviceOnly extends ReadMode
// object DeviceFirst extends ReadMode

case class HybridParquetOpts(mode: String,
                             maxConcurrent: Int,
                             batchSizeBytes: Long,
                             pollInterval: Int,
                             maxPreloadBytes: Long,
                             async: Boolean)
