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

import java.util.{Optional, TimeZone}
import java.util.concurrent.{LinkedBlockingQueue, Semaphore, TimeUnit}
import java.util.concurrent.atomic.{AtomicBoolean, AtomicInteger}
import java.util.concurrent.locks.ReentrantLock

import scala.collection.JavaConverters._
import scala.collection.mutable

import ai.rapids.cudf.{HostColumnVector, HostMemoryBuffer, NvtxColor, Table}
import com.nvidia.spark.rapids.{DateTimeRebaseMode, GpuColumnVector, GpuMetric, GpuSemaphore, HMBInputFile, NvtxWithMetrics}
import com.nvidia.spark.rapids.Arm.{closeOnExcept, withResource}
import com.nvidia.spark.rapids.RapidsPluginImplicits._
import com.nvidia.spark.rapids.RmmRapidsRetryIterator.{AutoCloseableAttemptSpliterator, RmmRapidsRetryAutoCloseableIterator}
import org.apache.hadoop.conf.Configuration
import org.apache.parquet.{HadoopReadOptions, VersionParser}
import org.apache.parquet.VersionParser.ParsedVersion
import org.apache.parquet.bytes.DirectByteBufferAllocator
import org.apache.parquet.column.page.PageReadStore
import org.apache.parquet.hadoop.ParquetFileReader
import org.apache.parquet.schema.MessageType
import org.apache.parquet.schema.PrimitiveType.PrimitiveTypeName

import org.apache.spark.TaskContext
import org.apache.spark.internal.Logging
import org.apache.spark.sql.execution.vectorized.rapids.RapidsWritableColumnVector
import org.apache.spark.sql.rapids.ComputeThreadPool
import org.apache.spark.sql.rapids.ComputeThreadPool.TaskWithPriority
import org.apache.spark.sql.types.{ArrayType, BinaryType, DataType, DecimalType, IntegerType, MapType, StringType, StructField, StructType}
import org.apache.spark.sql.vectorized.ColumnarBatch

case class AsyncBatchResult(data: Array[HostColumnVector],
                            sizeInByte: Long,
                            rowGroupInfo: Option[RowGroupInfo],
                            dictInfo: Option[HostDictionaryInfo]) extends AutoCloseable {
  override def close(): Unit = {
    data.safeClose()
    // DictionaryInfo is task-level global context. We only close the DictVector after finishing
    // all parquet reading work.
    // dictInfo.foreach(_.vectors.map(_._2).safeClose())
  }
}

class AsyncParquetReaderError(ex: Throwable) extends RuntimeException(ex)

class AsyncParquetReader(conf: Configuration,
                         tgtBatchSize: Int,
                         fileBuffer: HostMemoryBuffer,
                         offset: Long,
                         len: Long,
                         metrics: Map[String, GpuMetric],
                         dateRebaseMode: DateTimeRebaseMode,
                         timestampRebaseMode: DateTimeRebaseMode,
                         clippedSchema: MessageType,
                         slotAcquired: Boolean,
                         enableDictLateMat: Boolean,
                         asynchronous: Boolean,
                         directBuffering: Boolean)
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
    val reader = if (!directBuffering) {
      val options = HadoopReadOptions.builder(conf)
        .withRange(offset, offset + len)
        .withAllocator(new DirectByteBufferAllocator)
        .withCodecFactory(new ParquetHeapCodecFactory(conf, 0))
        .build()
      val bufferFile = new HMBInputFile(fileBuffer, length = Some(offset + len))
      new ParquetFileReader(bufferFile, options)
    } else {
      val options = HadoopReadOptions.builder(conf)
        .withRange(offset, offset + len)
        .withAllocator(new DirectByteBufferAllocator)
        .withCodecFactory(new ParquetDirectCodecFactory(conf, 0))
        .build()
      val bufferFile = new HMBInputFile(fileBuffer, length = Some(offset + len))
      new ParquetFileReader(bufferFile, options)
    }

    // The fileSchema here has already been clipped
    reader.setRequestedSchema(clippedSchema)
    reader
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
    var curRowGroup: PageReadStore = pageReader.readNextFilteredRowGroup()
    while (curRowGroup != null) {
      rowGroups += curRowGroup;
      curRowGroup = pageReader.readNextFilteredRowGroup()
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
    logInfo(s"[$taskID] rowCount:$totalRowCnt; rowBatchSize:$rowBatchSize; numBatches:$value")
    value
  }
  private val consumedBatches = new AtomicInteger(0)
  private val producedBatches = new AtomicInteger(0)
  // RowGroupOffset for Dictionary Push down.
  // Being initialized with -1, because the increment is ahead of the usage.
  @volatile private var rowGroupOffset = -1

  private val parquetColumn: ParquetColumn = {
    val converter = new ParquetToSparkSchemaConverter()
    converter.convertParquetColumn(clippedSchema, None)
  }

  private lazy val stringColumnMeta: Map[String, BinaryColumnMetaSummary] = {
    val mapBuilder = mutable.ArrayBuffer[(String, BinaryColumnMetaSummary)]()
    val stack = mutable.Stack[ParquetColumn](parquetColumn.children: _*)
    while (stack.nonEmpty) {
      stack.pop match {
        case pc if pc.isPrimitive =>
          val descriptor = pc.descriptor.get
          if (descriptor.getPrimitiveType.getPrimitiveTypeName == PrimitiveTypeName.BINARY) {
            mapBuilder += pc.path.mkString(".") ->
              PageMetaUtils.inspectColumn(rowGroupQueue, descriptor)
          }
        case pc =>
          pc.children.foreach(stack.push)
      }
    }
    mapBuilder.toMap
  }

  // compute while assembling dictionary columns
  // column offsets for the layout of flattened leaf columns
  private var parquetLeafOffsets: Array[Int] = _
  // indicate whether a top-level field contains DictLateMaterialize conversion or not
  private var hasDctLatMat: Array[Boolean] = _

  private lazy val dictColumns: mutable.TreeMap[Int, (HostColumnVector, Array[Int])] = {
    if (!enableDictLateMat) {
      mutable.TreeMap.empty[Int, (HostColumnVector, Array[Int])]
    } else {
      logDebug(s"[$taskID] Row Group size ${rowGroupQueue.size}")

      // val fieldNames = parquetColumn.sparkType.asInstanceOf[StructType].fields.map(_.name)
      val builder = mutable.TreeMap.empty[Int, (HostColumnVector, Array[Int])]
      val stack = mutable.Stack[ParquetColumn]()
      val leafOffsetsBuffer = mutable.ArrayBuffer[Int](0)
      val fieldHasDctLatMat = mutable.ArrayBuffer[Boolean]()
      var leafIndex = 0

      parquetColumn.children.foreach { column =>
        var hasDctLatMat = false
        stack.push(column)
        while (stack.nonEmpty) {
          stack.pop() match {
            case c if c.isPrimitive =>
              stringColumnMeta.get(c.path.mkString(".")) match {
                case Some(binColMeta) if binColMeta.isAllDictEncoded =>
                  // TODO: refactor DictLateMat to support the scenario of multiple subColumns
                  if (hasDctLatMat) {
                    logError(s"Can NOT apply DictLatMat on ${c.path.mkString(".")} because " +
                      "there exists another subColumn on which DictLatMat are applied. " +
                      "Currently, DictLatMat can only apply once for each field")
                  } else {
                    val patch = DictLateMatUtils.buildDictLateMatPatch(
                      binColMeta.dictPages.get, c.descriptor.get)
                    builder.put(leafIndex,
                      patch.dictVector -> patch.dictPageOffsets)
                    hasDctLatMat = true
                    logInfo(s"[$taskID] Column ${c.path.mkString(".")} is all dictEncoded")
                  }
                case _ =>
              }
              leafIndex += 1
            case cv =>
              cv.children.reverseIterator.foreach(stack.push)
          }
        }
        leafOffsetsBuffer += leafIndex
        fieldHasDctLatMat += hasDctLatMat
      }

      if (builder.nonEmpty) {
        parquetLeafOffsets = leafOffsetsBuffer.toArray
      }
      hasDctLatMat = fieldHasDctLatMat.toArray

      builder
    }
  }

  private lazy val dataColumnBuilders: Array[RapidsWritableColumnVector] = {
    sparkTypesWithDict.fields.map {
      case f: StructField if f.dataType == StringType && stringColumnMeta.contains(f.name) =>
        val sizeInBytes = stringColumnMeta(f.name).sizeInBytes
        // The estimated char sizes are set to be a little larger than the exact size,
        // because if the initial allocation turns out to be not enough, the buffer growth
        // would be considerably expensive.
        val (batchRowSize, batchCharSize) = if (rowBatchSize >= totalRowCnt) {
          totalRowCnt -> sizeInBytes.toInt
        } else {
          rowBatchSize -> (rowBatchSize.toDouble / totalRowCnt * sizeInBytes).toInt
        }
        if (batchCharSize >= batchRowSize) {
          logInfo(s"[$taskID] Initialize StringColBuffer(total byteSize $sizeInBytes) " +
            s"${f.name} with $batchRowSize rows and $batchCharSize bytes")
          new RapidsWritableColumnVector(batchRowSize, f.dataType, Optional.of(batchCharSize))
        } else {
          logInfo(s"[$taskID] can NOT initialize StringColBuffer(total byteSize $sizeInBytes) " +
            s"${f.name}  $batchRowSize rows * 4 > $batchCharSize bytes")
          new RapidsWritableColumnVector(batchRowSize, f.dataType)
        }
      case f: StructField =>
        new RapidsWritableColumnVector(rowBatchSize min totalRowCnt, f.dataType)
    }
  }

  private lazy val sparkTypesWithDict: StructType = {
    if (dictColumns.isEmpty) {
      parquetColumn.sparkType.asInstanceOf[StructType]
    } else {
      val emptyBuf = () => mutable.ArrayBuffer[StructField]()

      var leafIndex = 0
      val rootBuffer = emptyBuf()
      val stack = mutable.Stack[(StructField,
        mutable.ArrayBuffer[StructField], mutable.ArrayBuffer[StructField])](
        (StructField("", parquetColumn.sparkType, nullable = false), rootBuffer, emptyBuf()))

      while (stack.nonEmpty) {
        val (field, parentBuffer, childBuffer) = stack.head
        field.dataType match {
          case st: StructType =>
            if (st.fields.isEmpty || childBuffer.nonEmpty) {
              parentBuffer += StructField(field.name, StructType(childBuffer), field.nullable)
              stack.pop()
            } else {
              st.fields.reverseIterator.foreach(f => stack.push((f, childBuffer, emptyBuf())))
            }
          case at: ArrayType =>
            if (childBuffer.nonEmpty) {
              parentBuffer += StructField(field.name,
                ArrayType(childBuffer.head.dataType), field.nullable)
              stack.pop()
            } else {
              stack.push((StructField("", at.elementType, nullable = false),
                childBuffer, emptyBuf()))
            }
          case mt: MapType =>
            if (childBuffer.nonEmpty) {
              parentBuffer += StructField(field.name,
                MapType(childBuffer(0).dataType, childBuffer(1).dataType), field.nullable)
              stack.pop()
            } else {
              stack.push((StructField("", mt.valueType, nullable = false),
                childBuffer, emptyBuf()))
              stack.push((StructField("", mt.keyType, nullable = false),
                childBuffer, emptyBuf()))
            }
          case _ =>
            if (dictColumns.contains(leafIndex)) {
              parentBuffer += StructField(field.name, IntegerType, field.nullable)
            } else {
              parentBuffer += field
            }
            stack.pop()
            leafIndex += 1
        }
      }
      logDebug(s"[$taskID] DictLatMat: ${parquetColumn.sparkType} to ${rootBuffer.head.dataType}")
      rootBuffer.head.dataType.asInstanceOf[StructType]
    }
  }

  private lazy val columnVectors: Array[ParquetColumnVector] =  {
    initialized.getAndSet(true)

    dataColumnBuilders.indices.toArray.map { i =>
      new ParquetColumnVector(parquetColumn.children(i),
        dataColumnBuilders(i),
        rowBatchSize min totalRowCnt,
        Set.empty[ParquetColumn].asJava, true, -1, null,
        if (hasDctLatMat == null) false else hasDctLatMat(i),
      );
    }
  }

  private lazy val hostDictionaryInfo = {
    val (colIdx, dictData) = dictColumns.unzip
    val (dictVec, dictSliceOffsets) = dictData.unzip
    logInfo(s"[$taskID] column index: $colIdx -> " +
      s"dictSlices: [${dictSliceOffsets.map(_.mkString(",")).mkString(" | ")}]")
    HostDictionaryInfo(
      colIdx.toArray, dictVec.toArray, dictSliceOffsets.toArray,
      parquetLeafOffsets)
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
    val startRowGroup = if (currentGroup == null || remainPageRows == 0) {
      rowGroupOffset + 1
    } else {
      rowGroupOffset
    }
    // splitsOnLeaves is used to collect the split points over each RowGroup for each leaf column
    val leafCnt = columnVectors.foldLeft(0) { case (cnt, cv) => cnt + cv.getLeaves.size() }
    val splitsOnLeaves = (0 until leafCnt).map(_ => mutable.ArrayBuffer.empty[Int]).toArray

    while (remainBatchRows > 0) {
      if (currentGroup == null || remainPageRows == 0) {
        rowGroupOffset += 1
        currentGroup = rowGroupQueue.dequeue()
        remainPageRows = currentGroup.getRowCount.toInt
        // update column readers to read the new page
        val stack = mutable.Stack[ParquetColumnVector]()
        columnVectors.reverseIterator.foreach(stack.push)
        var primitiveColIndex = 0
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
                  writerVersion,
                  dictColumns.contains(primitiveColIndex)))
              primitiveColIndex += 1
            case cv =>
              cv.getChildren.asScala.reverseIterator.foreach(stack.push)
          }
        }
      }

      val readSize = remainBatchRows min remainPageRows
      remainPageRows -= readSize
      remainBatchRows -= readSize
      remainTotalRows -= readSize

      var leafIdx = 0
      columnVectors.foreach { cv =>
        cv.getLeaves.asScala.foreach {
          case leaf if leaf.getColumnReader != null =>
            leaf.getColumnReader.readBatch(readSize,
              leaf.getValueVector,
              leaf.getRepetitionLevelVector,
              leaf.getDefinitionLevelVector
            )
            // Collect split point of this RowGroup before being reset
            splitsOnLeaves(leafIdx) += leaf.getValueVector
              .asInstanceOf[RapidsWritableColumnVector]
              .getCurrentRowGroupSize
            leafIdx += 1
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
    val batchCols = dataColumnBuilders.map(_.build(sizeCounter))

    // update batch size and remaining
    remainBatchRows = rowBatchSize min remainTotalRows
    // Reset all the HostColumnBuffers for the upcoming batch
    dataColumnBuilders.foreach(_.reallocate(remainBatchRows))

    // Create DictInfo and RowGroupInfo only if DictLateMat will be applied on at least one column
    if (dictColumns.isEmpty) {
      AsyncBatchResult(batchCols, sizeCounter.head, None, None)
    } else {
      val rgInfo = {
        val rowGroupLength = rowGroupOffset - startRowGroup + 1
        require(splitsOnLeaves.forall(_.length == rowGroupLength),
          s"splitOnLeaves are not aligned with rowGroupLength($rowGroupLength) " +
            s"[${splitsOnLeaves.map(_.mkString("|")).mkString("] [")}]"
        )
        RowGroupInfo(startRowGroup, startRowGroup + rowGroupLength,
          splitsOnLeaves.map(_.toArray))
      }
      AsyncBatchResult(batchCols, sizeCounter.head, Some(rgInfo), Some(hostDictionaryInfo))
    }
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
            logDebug(s"[$taskID] produced a new batch($numProduced/$numBatches)")
          case Right(exception) =>
            throw exception
        }
      }
    }
  }

  private def launchTask(): Unit = if (asynchronous && task == null) {
    val failedCallback = new ComputeThreadPool.FailedCallback {
      private val queue = resultQueue
      override def callback(ex: Throwable): Unit = {
        queue.offer(Right(ex))
      }
    }
    task = new TaskWithPriority(() => readAsProducer(),
      0, slotAcquired, null, failedCallback)
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
          logDebug(s"[$taskID] consumed a new batch($numConsumed/$numBatches)")
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
            enableDictLateMat: Boolean,
            asynchronous: Boolean,
            directBuffering: Boolean): AsyncParquetReader = {
    new AsyncParquetReader(conf,
      tgtBatchSize, fileBuffer, offset, len,
      metrics,
      dateRebaseMode, timestampRebaseMode,
      clippedSchema,
      slotAcquired, enableDictLateMat, asynchronous, directBuffering)
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

    private var dictionaryInfo: Option[DeviceDictionaryInfo] = None
    private var hostDctVecHolder: Array[HostColumnVector] = Array.ofDim(0)

    def execute(batchResult: AsyncBatchResult): ColumnarBatch = {

      withResource(new NvtxWithMetrics(
        "GPU wait time before H2D", NvtxColor.WHITE, metrics("preH2dGpuWait"))) { _ =>
        GpuSemaphore.acquireIfNecessary(TaskContext.get())
      }

      if (dictionaryInfo.isEmpty && batchResult.dictInfo.nonEmpty) {
        val hInfo = batchResult.dictInfo.get
        hostDctVecHolder = hInfo.vectors
        val deviceDictVectors = hostDctVecHolder.safeMap(_.copyToDevice())
        dictionaryInfo = Some(DeviceDictionaryInfo(
          hInfo.columnIndices, deviceDictVectors, hInfo.dictSliceOffsets,
          hInfo.leafColumnOffsets))
      }

      val ret = withResource(new NvtxWithMetrics("Transfer HostVectors to Device", NvtxColor.CYAN,
        metrics("hostVecToDeviceTime"))) { _ =>

        // Do NOT close hostColumnVectors here in case of retry (it will be closed via
        // AsyncBatchResult::close)
        val deviceCVs = batchResult.data.safeMap(_.copyToDevice())

        metrics.get("cpuDecodeRows").foreach(_.+=(batchResult.data.head.getRowCount))
        metrics.get("cpuDecodeBatches").foreach(_.+=(1))
        metrics.get("numOutputBatches").foreach(_.+=(1))

        val dCVs = if (dictionaryInfo.isEmpty) {
          deviceCVs
        } else {
          closeOnExcept(deviceCVs) { _ =>
            DictLateMatUtils.decodeDictionary(
              deviceCVs, dictionaryInfo.get, batchResult.rowGroupInfo.get)
          }
        }
        withResource(new Table(dCVs: _*)) { table =>
          dCVs.safeClose()
          GpuColumnVector.from(table, dataTypes)
        }
      }

      ret
    }

    override def close(): Unit = {
      dictionaryInfo.foreach(_.vectors.safeClose())
      hostDctVecHolder.safeClose()
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
        initialized.set(true)
      }
    }
  }

  private var syncSemaphore: Semaphore = _
  private val initialized: AtomicBoolean = new AtomicBoolean(false)
}

case class HybridParquetOpts(mode: String,
                             maxConcurrent: Int,
                             batchSizeBytes: Long,
                             pollInterval: Int,
                             enableDictLateMat: Boolean,
                             async: Boolean,
                             unsafeDecompression: Boolean)
