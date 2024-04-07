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

import java.util.Optional

import scala.collection.mutable

import ai.rapids.cudf.{DType, HostColumnVector, HostColumnVectorCore, HostMemoryBuffer}
import org.apache.parquet.column.ColumnDescriptor
import org.apache.parquet.column.page.{DataPage, DataPageV1, DataPageV2, DictionaryPage, PageReader, PageReadStore}
import org.apache.parquet.column.values.dictionary.PlainValuesDictionary.PlainBinaryDictionary

import org.apache.spark.internal.Logging

case class DictLatMatInfo(dictVector: HostColumnVector, dictPageOffsets: Array[Int])

object DictLateMatUtils extends Logging {

  def extractDict(rowGroups: Seq[PageReadStore],
                  descriptor: ColumnDescriptor): Option[DictLatMatInfo] = {

    val dictPages = mutable.ArrayBuffer[DictionaryPage]()

    // Go through each RowGroup and each page inside them to check if all pages use Dictionary.
    // Dictionary late materialization only works if all pages use Dictionary.
    rowGroups.foreach { rowGroup =>
      val pageReader = rowGroup.getPageReader(descriptor)
      val dictPage = pageReader.readDictionaryPage()
      if (dictPage == null || !isAllDictEncoded(pageReader)) {
        return None
      }
      dictPages += dictPage
    }

    Some(combineDictPages(dictPages, descriptor))
  }

  private def combineDictPages(dictPages: Seq[DictionaryPage],
                               descriptor: ColumnDescriptor): DictLatMatInfo = {
    val pageOffsets = mutable.ArrayBuffer[Int](0)
    var rowNum: Int = 0

    val dictionaries = dictPages.map { dictPage =>
      val dictionary = dictPage.getEncoding.initDictionary(descriptor, dictPage)
        .asInstanceOf[PlainBinaryDictionary]
      rowNum += dictionary.getMaxId + 1
      pageOffsets += rowNum
      dictionary
    }

    var charNum: Int = 0
    val offsetBuf = HostMemoryBuffer.allocate((rowNum + 1) * 4L)
    offsetBuf.setInt(0, 0)
    var i = 1
    dictionaries.foreach { dict =>
      (0 to dict.getMaxId).foreach { j =>
        charNum += dict.decodeToBinary(j).length()
        offsetBuf.setInt(i * 4L, charNum)
        i += 1
      }
    }

    val charBuf = HostMemoryBuffer.allocate(charNum)
    i = 0
    dictionaries.foreach { dict =>
      (0 to dict.getMaxId).foreach { j =>
        val ba = dict.decodeToBinary(j).getBytes
        charBuf.setBytes(offsetBuf.getInt(i * 4L), ba, 0, ba.length)
        i += 1
      }
    }

    val dictVector = new HostColumnVector(DType.STRING, rowNum, Optional.of(0L),
      charBuf, null, offsetBuf, new java.util.ArrayList[HostColumnVectorCore]())

    DictLatMatInfo(dictVector, pageOffsets.toArray)
  }

  private def isAllDictEncoded(pageReader: PageReader): Boolean = {
    require(ccPageReader.isInstance(pageReader),
      "Only supports org.apache.parquet.hadoop.ColumnChunkPageReadStore.ColumnChunkPageReader")
    val rawPagesField = ccPageReader.getDeclaredField("compressedPages")
    rawPagesField.setAccessible(true)

    val pageQueue = rawPagesField.get(pageReader).asInstanceOf[java.util.ArrayDeque[DataPage]]
    val swapQueue = new java.util.ArrayDeque[DataPage]()
    var allDictEncoded = true

    while (!pageQueue.isEmpty) {
      swapQueue.addLast(pageQueue.pollFirst())
      if (allDictEncoded) {
        allDictEncoded = swapQueue.getLast match {
          case p: DataPageV1 =>
            p.getValueEncoding.usesDictionary()
          case p: DataPageV2 =>
            p.getDataEncoding.usesDictionary()
        }
      }
    }
    while (!swapQueue.isEmpty) {
      pageQueue.addLast(swapQueue.pollFirst())
    }

    allDictEncoded
  }

  private val ccPageReader: Class[_] = {
    val ccPageReadStore = Class.forName("org.apache.parquet.hadoop.ColumnChunkPageReadStore")
    val ccPageReader = ccPageReadStore.getDeclaredClasses.find { memberClz =>
      memberClz.getSimpleName.equals("ColumnChunkPageReader")
    }
    require(ccPageReader.nonEmpty, "can NOT find the Class definition of ColumnChunkPageReader")
    ccPageReader.get
  }

}
