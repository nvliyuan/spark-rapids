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

import ai.rapids.cudf.{ColumnVector, HostColumnVector, HostMemoryBuffer, NvtxColor, Scalar, Table}
import com.nvidia.spark.rapids.{DateTimeRebaseMode, GpuColumnVector, GpuMetric, GpuSemaphore, HMBInputFile, NvtxWithMetrics}
import com.nvidia.spark.rapids.Arm.{closeOnExcept, withResource}
import com.nvidia.spark.rapids.RapidsPluginImplicits._
import com.nvidia.spark.rapids.RmmRapidsRetryIterator.{AutoCloseableAttemptSpliterator, RmmRapidsRetryAutoCloseableIterator}
import org.apache.hadoop.conf.Configuration
import org.apache.parquet.{HadoopReadOptions, VersionParser}
import org.apache.parquet.VersionParser.ParsedVersion
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


case class HostDictionaryInfo(columnIndices: Array[Int],
                              vectors: Array[HostColumnVector],
                              dictSliceOffsets: Array[Array[Int]],
                              leafColumnOffsets: Array[Int],
                              rowGroupOffsets: Array[Int])

case class DeviceDictionaryInfo(columnIndices: Array[Int],
                                vectors: Array[ColumnVector],
                                dictSliceOffsets: Array[Array[Int]],
                                leafColumnOffsets: Array[Int],
                                rowGroupOffsets: Array[Int])

case class AsyncBatchResult(data: Array[HostColumnVector],
                            dictInfo: Option[HostDictionaryInfo],
                            globalRowOffset: Int,
                            sizeInRow: Int,
                            sizeInByte: Long) extends AutoCloseable {
  override def close(): Unit = {
    data.safeClose()
    // DictionaryInfo is task-level global context. We only close the DictVector after finishing
    // all parquet reading work.
    // dictInfo.foreach(_.vectors.map(_._2).safeClose())
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
    enableDictLateMat: Boolean,
    asynchronous: Boolean)
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
    logInfo(s"[$taskID] rowCount:$totalRowCnt; rowBatchSize:$rowBatchSize; numBatches:$value")
    value
  }
  private val consumedBatches = new AtomicInteger(0)
  private val producedBatches = new AtomicInteger(0)
  // global row offset for Dictionary Push down
  @volatile private var globalRowOffset = 0

  private val parquetColumn: ParquetColumn = {
    val converter = new ParquetToSparkSchemaConverter()
    converter.convertParquetColumn(clippedSchema, None)
  }

  private lazy val topLevelStringColumnMeta: Map[String, BinaryColumnMetaSummary] = {
    val fieldNames = parquetColumn.sparkType.asInstanceOf[StructType].fields.map(_.name)
    val mapBuilder = mutable.ArrayBuffer[(String, BinaryColumnMetaSummary)]()
    parquetColumn.children.zipWithIndex.foreach {
      case (c, i) if c.isPrimitive =>
        val descriptor = c.descriptor.get
        if (descriptor.getPrimitiveType.getPrimitiveTypeName == PrimitiveTypeName.BINARY) {
          mapBuilder += fieldNames(i) ->
            BinaryColumnMetaUtils.inspectColumn(rowGroupQueue, descriptor)
        }
      case _ =>
    }
    mapBuilder.toMap
  }

  // compute while assembling dictionary columns
  // column offsets for the layout of flattened leaf columns
  private var parquetLeafOffsets: Array[Int] = _
  // row offsets of each row group
  private var rowGroupRowOffsets: Array[Int] = _
  // indicate whether a top-level field contains DictLateMaterialize conversion or not
  private var hasDctLatMat: Array[Boolean] = _

  private lazy val dictColumns: mutable.TreeMap[Int, (HostColumnVector, Array[Int])] = {
    if (!enableDictLateMat) {
      mutable.TreeMap.empty[Int, (HostColumnVector, Array[Int])]
    } else {
      logInfo(s"[$taskID] Row Group size ${rowGroupQueue.size}")

      val fieldNames = parquetColumn.sparkType.asInstanceOf[StructType].fields.map(_.name)
      val builder = mutable.TreeMap.empty[Int, (HostColumnVector, Array[Int])]
      val stack = mutable.Stack[ParquetColumn]()
      val leafOffsetsBuffer = mutable.ArrayBuffer[Int](0)
      val fieldHasDctLatMat = mutable.ArrayBuffer[Boolean]()
      var leafIndex = 0

      parquetColumn.children.zipWithIndex.foreach { case (column, fieldIndex) =>
        val numDictFound = builder.size
        stack.push(column)
        while (stack.nonEmpty) {
          stack.pop() match {
            // TODO: support Dictionary late materialize on columns of nested type
            case c if c.isPrimitive =>
              topLevelStringColumnMeta.get(fieldNames(fieldIndex)) match {
                case Some(binColMeta) if binColMeta.isAllDictEncoded =>
                  val patch = BinaryColumnMetaUtils.buildDictLateMatPatch(
                    binColMeta.dictPages.get, c.descriptor.get)
                  builder.put(leafIndex,
                    patch.dictVector -> patch.dictPageOffsets)

                  // Compute rowGroupRowOffsets if DictLateMaterialize will be applied on
                  // at least one column.
                  if (rowGroupRowOffsets == null) {
                    val rgRowOff = Array.ofDim[Int](rowGroupQueue.size + 1)
                    rowGroupQueue.indices.foreach { i =>
                      rgRowOff(i + 1) = rgRowOff(i) + rowGroupQueue(i).getRowCount.toInt
                    }
                    rowGroupRowOffsets = rgRowOff
                  }
                  logInfo(s"[$taskID] Column ${c.path.mkString(".")} is all dictEncoded")
                case _ =>
              }
              leafIndex += 1
            case cv =>
              cv.children.reverseIterator.foreach(stack.push)
          }
        }
        leafOffsetsBuffer += leafIndex
        fieldHasDctLatMat += builder.size > numDictFound
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
      case f: StructField if f.dataType == StringType
        && topLevelStringColumnMeta.contains(f.name) =>
        val sizeInBytes = topLevelStringColumnMeta(f.name).sizeInBytes
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
            s"${f.name}  $batchRowSize rows > $batchCharSize bytes")
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
    val batchCols = dataColumnBuilders.map(_.build(sizeCounter))
    // update batch size and remaining
    remainBatchRows = rowBatchSize min remainTotalRows
    // Reset all the HostColumnBuffers for the upcoming batch
    dataColumnBuilders.foreach(_.reallocate(remainBatchRows))

    // Only return DictionaryInfo on the first batch since it works as global context
    val dictInfo = if (globalRowOffset > 0 || dictColumns.isEmpty) {
      None
    } else {
      val (colIdx, dictData) = dictColumns.unzip
      val (dictVec, dictSliceOffsets) = dictData.unzip
      logDebug(s"[$taskID] column index: $colIdx -> " +
        s"dictSlices: [${dictSliceOffsets.map(_.mkString(",")).mkString(" | ")}]")
      Some(HostDictionaryInfo(
        colIdx.toArray, dictVec.toArray, dictSliceOffsets.toArray,
        parquetLeafOffsets, rowGroupRowOffsets))
    }

    val batchRowSize = batchCols.head.getRowCount.toInt
    val batchRowOffset = globalRowOffset
    globalRowOffset += batchRowSize
    AsyncBatchResult(batchCols, dictInfo, batchRowOffset, batchRowSize, sizeCounter.head)
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
            asynchronous: Boolean): AsyncParquetReader = {
    new AsyncParquetReader(conf,
      tgtBatchSize, fileBuffer, offset, len,
      metrics,
      dateRebaseMode, timestampRebaseMode,
      clippedSchema,
      slotAcquired, enableDictLateMat, asynchronous)
  }

  def decodeDictionary(columns: Array[ColumnVector],
                       rowStart: Int,
                       rowEnd: Int,
                       dictInfo: DeviceDictionaryInfo): Array[ColumnVector] = {

    // 1. compute the slice index range of current batch
    // 2. compute the row size of each range partition if the range crossing multiple RowGroups
    val (sliceStart, sliceEnd, slicePartSizes) = {
      val rgOff: Array[Int] = dictInfo.rowGroupOffsets
      var start = -1
      var end = 0
      while (rgOff(end + 1) < rowEnd) {
        if (start < 0 && rgOff(end + 1) > rowStart) {
          start = end
        }
        end += 1
      }
      if (start < 0) start = end

      val partSizes = mutable.ArrayBuffer[Int]()
      if (start < end) {
        partSizes += rgOff(start + 1) - rowStart
        ((start + 1) until end).foreach { i =>
          partSizes += rgOff(i + 1) - rgOff(i)
        }
        partSizes += rowEnd - rgOff(end)
      }

      (start, end, partSizes.toArray)
    }

    val leafOffsets: Array[Int] = dictInfo.leafColumnOffsets
    val dictSliceInfo: Array[Array[Int]] = dictInfo.dictSliceOffsets
    var fieldIdx: Int = 0
    var dictVecIdx: Int = 0
    dictInfo.columnIndices.zip(dictInfo.vectors).foreach { case (leafIdx, dictVector) =>
      closeOnExcept(dictVector) { _ =>
        while (fieldIdx < columns.length && leafIdx >= leafOffsets(fieldIdx + 1)) {
          fieldIdx += 1
        }
        if (fieldIdx == columns.length || leafOffsets(fieldIdx) != leafIdx) {
          throw new UnsupportedOperationException(
            "Currently only supports dictionary decoding for non-nested string columns")
        }
      }

      // replace null indices with OUT_OF_BOUND value
      val dictSize = dictVector.getRowCount.toInt
      val localIndexVec = withResource(columns(fieldIdx)) { col =>
        if (!col.hasNulls) {
          col.incRefCount()
        } else {
          withResource(Scalar.fromInt(dictSize)) { oobVal =>
            col.replaceNulls(oobVal)
          }
        }
      }

      /* // check correctness is DictIndices
      val slice = dictSliceInfo(dictVecIdx)
      val ubVal = (0 until slice.length - 1).map(i => slice(i + 1) - slice(i)).max
      withResource(localIndexVec.max()) { idxMax =>
        require(idxMax.getInt <= ubVal, s"index ${idxMax.getInt} out of bound $ubVal")
      } */

      // convert the local indexVector to the global indexVector (across multiple DictPages)
      val indexVector = withResource(localIndexVec) { col =>
        if (sliceStart == sliceEnd) {
          // set the same offset since current batch does not cross RowGroups
          withResource(Scalar.fromInt(dictSliceInfo(dictVecIdx)(sliceEnd))) { s =>
            col.add(s)
          }
        } else {
          //
          val partCVs = withResource(mutable.ArrayBuffer[Scalar]()) { scalars =>
            (sliceStart to sliceEnd).foreach { i =>
              scalars += Scalar.fromInt(dictSliceInfo(dictVecIdx)(i))
            }
            scalars.zip(slicePartSizes).safeMap { case (s, partLen) =>
              ColumnVector.fromScalar(s, partLen)
            }
          }
          val offsetVec = withResource(partCVs) { _ =>
            ColumnVector.concatenate(partCVs: _*)
          }
          withResource(offsetVec) { _ =>
            col.add(offsetVec)
          }
        }
      }
      dictVecIdx += 1

      // Decode the BinaryDictionary with indexVector and replace the original column
      columns(fieldIdx) = withResource(indexVector) { _ =>
        withResource(new Table(dictVector)) { dictTable =>
          dictTable.gather(indexVector).getColumn(0)
        }
      }
    }
    columns
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
          hInfo.leafColumnOffsets, hInfo.rowGroupOffsets))
      }

      val ret = withResource(new NvtxWithMetrics("Transfer HostVectors to Device", NvtxColor.CYAN,
        metrics("hostVecToDeviceTime"))) { _ =>

        // Do NOT close hostColumnVectors here in case of retry (it will be closed via
        // AsyncBatchResult::close)
        val deviceCVs = batchResult.data.safeMap(_.copyToDevice())

        metrics.get("cpuDecodeRows").foreach(_.+=(batchResult.sizeInRow))
        metrics.get("cpuDecodeBatches").foreach(_.+=(1))
        metrics.get("numOutputBatches").foreach(_.+=(1))

        val dCVs = if (dictionaryInfo.isEmpty) {
          deviceCVs
        } else {
          closeOnExcept(deviceCVs) { _ =>
            AsyncParquetReader.decodeDictionary(
              deviceCVs,
              batchResult.globalRowOffset,
              batchResult.globalRowOffset + batchResult.sizeInRow,
              dictionaryInfo.get)
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
                             async: Boolean)
