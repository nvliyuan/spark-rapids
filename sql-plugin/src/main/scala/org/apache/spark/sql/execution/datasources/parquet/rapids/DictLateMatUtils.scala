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

import java.util
import java.util.Optional

import scala.collection.JavaConverters._
import scala.collection.mutable

import ai.rapids.cudf._
import com.nvidia.spark.rapids.Arm.withResource
import com.nvidia.spark.rapids.RapidsPluginImplicits._
import org.apache.parquet.column.ColumnDescriptor
import org.apache.parquet.column.page.DictionaryPage
import org.apache.parquet.column.values.dictionary.PlainValuesDictionary.PlainBinaryDictionary

import org.apache.spark.internal.Logging


case class DictLatMatPatch(dictVector: HostColumnVector, dictPageOffsets: Array[Int])

case class HostDictionaryInfo(columnIndices: Array[Int],
                              vectors: Array[HostColumnVector],
                              dictSliceOffsets: Array[Array[Int]],
                              leafColumnOffsets: Array[Int])

case class DeviceDictionaryInfo(columnIndices: Array[Int],
                                vectors: Array[ColumnVector],
                                dictSliceOffsets: Array[Array[Int]],
                                leafColumnOffsets: Array[Int])

case class RowGroupInfo(startGroup: Int, endGroup: Int, splitsOnLeaves: Array[Array[Int]])


object DictLateMatUtils extends Logging {

  def buildDictLateMatPatch(dictPages: Seq[DictionaryPage],
                            descriptor: ColumnDescriptor): DictLatMatPatch = {
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
    // There exists dict without any non-empty string values, in case of null ptr error during
    // copyFromHostToDevice, allocating at least 1 byte for char buffer.
    val charBuf = HostMemoryBuffer.allocate(charNum max 1)
    i = 0
    dictionaries.foreach { dict =>
      (0 to dict.getMaxId).foreach { j =>
        val ba = dict.decodeToBinary(j).getBytes
        charBuf.setBytes(offsetBuf.getInt(i * 4L), ba, 0, ba.length)
        i += 1
      }
    }

    val dictVector = new HostColumnVector(DType.STRING, rowNum, Optional.of(0L),
      charBuf, null, offsetBuf, new util.ArrayList[HostColumnVectorCore]())
    logError(s"Built the HostDictVector for Column(${descriptor.getPath.mkString(".")}): " +
      s"${dictVector.getRowCount}rows/${charBuf.getLength}bytes")

    DictLatMatPatch(dictVector, pageOffsets.toArray)
  }

  def decodeDictionary(columns: Array[ColumnVector],
                       dictInfo: DeviceDictionaryInfo,
                       rgInfo: RowGroupInfo): Array[ColumnVector] = {

    val leafOffsets: Array[Int] = dictInfo.leafColumnOffsets
    val dictSliceInfo: Array[Array[Int]] = dictInfo.dictSliceOffsets
    var fieldIdx: Int = 0

    dictInfo.vectors.indices.foreach { dictIdx =>
      val dictLeafIdx = dictInfo.columnIndices(dictIdx)
      val dictVector = dictInfo.vectors(dictIdx)

      while (fieldIdx < columns.length && dictLeafIdx >= leafOffsets(fieldIdx + 1)) {
        fieldIdx += 1
      }
      val localLeafIndex = dictLeafIdx - leafOffsets(fieldIdx)
      // We need to build a helper structure to traverse ColumnTreeHierarchy
      val colTreeView = ColumnTreeView(columns(fieldIdx))
      val localIndexVec = colTreeView.getLeafColumnView(localLeafIndex)

      // Convert local index to the global index
      val globalIndexVec = convertLocalIndexToGlobal(localIndexVec,
        dictSliceInfo(dictIdx),
        rgInfo.startGroup -> rgInfo.endGroup,
        rgInfo.splitsOnLeaves(dictLeafIdx))

      // Decode the BinaryDictionary with indexVector and replace the original column
      val materialized = withResource(new Table(dictVector)) { dictTable =>
        withResource(globalIndexVec) { indexCol =>
          dictTable.gather(indexCol).getColumn(0)
        }
      }

      // Inject the materialized string column into the position of corresponding index column.
      // And the rebuild the entire column hierarchy recursively.
      // The reason of releasing the original fieldCol and the indexCol after building is that
      // ColumnTreeView takes the ownership of underlying buffers during building the replaced
      // vector (by adding refCount).
      columns(fieldIdx) = withResource(materialized) { _ =>
        colTreeView.replaceLeafNode(localLeafIndex, materialized)
        colTreeView.buildColumnVector()
      }
    }

    columns
  }

  private def convertLocalIndexToGlobal(localIdxVec: ColumnView,
                                        dictSliceOffsets: Array[Int],
                                        rowGroupRange: (Int, Int),
                                        rowGroupSplits: Array[Int]): ColumnVector = {
    // replace null indices with OUT_OF_BOUND value
    val dictSize = dictSliceOffsets.last
    val indexVec = withResource(Scalar.fromInt(dictSize)) { oobVal =>
      localIdxVec.replaceNulls(oobVal)
    }

    // check correctness is DictIndices
    // val slice = dictSliceInfo(dictVecIdx)
    // val ubVal = (0 until slice.length - 1).map(i => slice(i + 1) - slice(i)).max
    // withResource(localIndexVec.max()) { idxMax =>
    //   require(idxMax.getInt <= ubVal, s"index ${idxMax.getInt} out of bound $ubVal")
    // }

    // convert the local indexVector to the global indexVector (across multiple DictPages)
    val (rgStart, rgEnd) = rowGroupRange
    withResource(indexVec) { col =>
      if (rgStart == rgEnd - 1) {
        if (dictSliceOffsets(rgStart) == 0) {
          // local_index == global_index if offset is all zeros
          col.incRefCount()
        } else {
          // set the same offset since current batch does not cross RowGroups
          withResource(Scalar.fromInt(dictSliceOffsets(rgStart))) { s =>
            col.add(s)
          }
        }
      } else {
        // set different offsets for different RowGroups, then concat them.
        val partCVs = withResource(mutable.ArrayBuffer[Scalar]()) { scalars =>
          (rgStart until  rgEnd).foreach { i =>
            scalars += Scalar.fromInt(dictSliceOffsets(i))
          }
          scalars.zip(rowGroupSplits).safeMap { case (s, partLen) =>
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
  }

}

private case class ColTreeNode(var view: ColumnView,
                               flatIndex: Int,
                               parent: Int,
                               var leafOnlyIndex: Int = -1,
                               var bufferRange: (Int, Int) = (-1, -1),
                               // bufferAddr is just used for debug
                               val bufferAddr: mutable.Buffer[Long] = mutable.Buffer[Long]()) {
  val children: Array[Int] = Array.ofDim(view.getNumChildren)
  var unFilledChildNum: Int = children.length
}

class ColumnTreeView(vector: ColumnVector,
                     nodes: Array[ColTreeNode],
                     buffers: Array[DeviceMemoryBuffer]) extends Logging {

  private val leafIdxToIdx: Map[Int, Int] = {
    nodes.filter(_.leafOnlyIndex > -1)
      .map(n => n.leafOnlyIndex -> n.flatIndex).toMap
  }
  // logError(s"$leafIdxToIdx \n nodes: ${nodes.toList} \n buffers: ${buffers.mkString(" | ")}")
  private var built: Boolean = false
  private var replaced: Boolean = false
  private var replacedIndex: Int = -1

  def getLeafColumnView(leafOnlyIndex: Int): ColumnView = {
    require(leafIdxToIdx.contains(leafOnlyIndex), s"Illegal leaf index $leafOnlyIndex")
    nodes(leafIdxToIdx(leafOnlyIndex)).view
  }

  def replaceLeafNode(leafOnlyIndex: Int, replaceVec: ColumnVector): Unit = {
    require(!replaced, "Only supports single replacement")
    replaced = true

    require(replaceVec.getNumChildren == 0, "The replaceVec is not leaf one")
    require(leafIdxToIdx.contains(leafOnlyIndex), s"Illegal leaf index $leafOnlyIndex")
    replacedIndex = leafIdxToIdx(leafOnlyIndex)

    // Dereference replaced DeviceMemoryBuffers. They will be closed by caller.
    val (start, end) = nodes(replacedIndex).bufferRange
    (start until end).foreach(i => buffers(i) = null)

    // replace the ColumnView
    nodes(replacedIndex).view.close()
    // Add refCount because the replaced vector will be closed along with another tentative
    // ColumnViews during building the result vector in bottom-up way.
    replaceVec.incRefCount()
    nodes(replacedIndex).view = replaceVec
  }

  def buildColumnVector(): ColumnVector = {
    require(replaced)
    require(!built)
    built = true

    val stack = mutable.Stack[Int](0)
    while (stack.nonEmpty) {
      val node = nodes(stack.top)
      if (node.unFilledChildNum == 0) {
        if (node.parent > -1) {
          nodes(node.parent).unFilledChildNum -= 1
          if (node.children.nonEmpty) {
            val curView = node.view
            val (validBuf, offsetsBuf) = {
              if (node.view.getType == DType.LIST) {
                node.bufferRange match {
                  case (st, ed) if ed - st == 0 =>
                    (None, None)
                  case (st, ed) if ed - st == 1 =>
                    (None, Some(buffers(st)))
                  case (st, ed) if ed - st == 2 =>
                    (Some(buffers(st)), Some(buffers(st + 1)))
                  case _ =>
                    throw new IllegalArgumentException(s"unmatched bufferRange for $node")
                }
              } else {
                // Either ListType or StructType (StringColumn has no child in terms of Java)
                val (st, ed) = node.bufferRange
                require(ed - st <= 1, s"unmatched bufferRange for $node")
                (if (ed == st) None else Some(buffers(st)), None)
              }
            }
            val childViews: Array[ColumnView] = node.children.map(i => nodes(i).view)
            node.view = new ColumnView(
              curView.getType, curView.getRowCount,
              Optional.of[java.lang.Long](curView.getNullCount),
              validBuf.orNull, offsetsBuf.orNull,
              childViews
            )
            // Collect all newly created ColumnViews, because `makeCudfColumnView` will copy
            // all children's view when building the parent view
            curView.close()
            childViews.safeClose()
          }
        }
        stack.pop()
      } else {
        node.children.reverseIterator.foreach(stack.push)
      }
    }

    // add reference count for DeviceBuffers since they will be referred by another Column
    buffers.foreach {
      case buf if buf == null =>
      case buf => buf.incRefCount()
    }

    // If the rootVector has been replaced, simply returns root.
    if (replacedIndex == 0) {
      return nodes.head.view.asInstanceOf[ColumnVector]
    }

    // partition top-level buffers and children buffers
    var childBufferOffset = 0
    val collectTopLevelBuffer = (bb: BaseDeviceMemoryBuffer) => if (bb != null) {
      childBufferOffset += 1
      buffers(childBufferOffset - 1)
    } else {
      null
    }
    val root = nodes.head.view
    val dataBuf = collectTopLevelBuffer(root.getData)
    val validBuf = collectTopLevelBuffer(root.getValid)
    val offsetsBuf = collectTopLevelBuffer(root.getOffsets)
    val childBuffers = new java.util.ArrayList[DeviceMemoryBuffer]()
    (childBufferOffset until buffers.length).foreach {
      case i if buffers(i) == null =>
      case i => childBuffers.add(buffers(i))
    }
    // The original (root) columnVector can be closed after the replacement are built
    val ret = withResource(vector) { _ =>
      new ColumnVector(
        root.getType,
        root.getRowCount, Optional.of(root.getNullCount),
        dataBuf, validBuf, offsetsBuf, childBuffers,
        nodes.head.children.map(i => nodes(i).view.getNativeView))
    }
    // direct children of root has not been cleaned up yet
    nodes.head.children.foreach(i => nodes(i).view.close())

    // Move the ownership of underlying rmm::device_buffer
    val replacedVector = nodes(replacedIndex).view.asInstanceOf[ColumnVector]
    ColumnTreeView.moveColumnHandle(replacedVector, ret)

    ret
  }

}

object ColumnTreeView {

  def apply(rootVec: ColumnVector): ColumnTreeView = {

    val allBuffers = extractMemoryBuffers(rootVec)

    var bufferCursor = 0
    val updateBufferInfo = (node: ColTreeNode) => {
      val start = bufferCursor
      node.bufferAddr += getDataAddr
        .invoke(null, node.view.getNativeView.asInstanceOf[java.lang.Long])
        .asInstanceOf[java.lang.Long]
      node.bufferAddr += getValidAddr
        .invoke(null, node.view.getNativeView.asInstanceOf[java.lang.Long])
        .asInstanceOf[java.lang.Long]
      node.bufferAddr += getOffsetAddr
        .invoke(null, node.view.getNativeView.asInstanceOf[java.lang.Long])
        .asInstanceOf[java.lang.Long]
      bufferCursor += node.bufferAddr.count(_ > 0)
      node.bufferRange = (start, bufferCursor)
    }

    if (rootVec.getNumChildren == 0) {
      val node = ColTreeNode(rootVec, 0, -1, 0)
      updateBufferInfo(node)
      return new ColumnTreeView(rootVec, Array(node), allBuffers)
    }

    val rootNode = ColTreeNode(rootVec, 0, -1, -1)
    updateBufferInfo(rootNode)
    var flatCursor = 1
    var leafOnlyCursor = 0
    val nodes = mutable.ArrayBuffer[ColTreeNode](rootNode)

    val stack = mutable.Stack[(ColTreeNode, Boolean)](rootNode -> false)
    // tmpStack is used to reverse the children order of entering stack
    val tmpStack = mutable.Stack[(ColTreeNode, Boolean)]()
    while (stack.nonEmpty) {
      val (node, visited) = stack.pop
      if (!visited && node.children.nonEmpty) {
        // push the popped node back as a visited item
        stack.push((node, true))
        // enroll children nodes
        var i = 0
        node.view.getChildColumnViews.foreach { childView =>
          val childNode = ColTreeNode(childView, flatCursor + i, node.flatIndex)
          nodes += childNode
          tmpStack.push((childNode, false))
          node.children(i) = flatCursor + i
          i += 1
        }
        while (tmpStack.nonEmpty) {
          stack.push(tmpStack.pop())
        }
        flatCursor += i
      }
      // BufferRange of rootVec is special and has already been set. So, only set for non-RootVec
      else if (stack.nonEmpty) {
        if (node.children.isEmpty) {
          node.leafOnlyIndex = leafOnlyCursor
          leafOnlyCursor += 1
        }
        updateBufferInfo(node)
      }
    }

    require(allBuffers.length == bufferCursor,
      s"nodes: ${nodes.toList} \n ${allBuffers.length}_buffers: ${allBuffers.mkString(" | ")}")

    new ColumnTreeView(rootVec, nodes.toArray, allBuffers)
  }

  private def extractMemoryBuffers(cv: ColumnVector): Array[DeviceMemoryBuffer] = {
    val offHeapState = offHeapField.get(cv.asInstanceOf[ColumnView])
    val toCloseField = offHeapState.getClass.getDeclaredField("toClose")
    toCloseField.setAccessible(true)
    val toClose = toCloseField.get(offHeapState).asInstanceOf[java.util.List[MemoryBuffer]]

    // Filter empty buffers, which can be NOT distinguished from `null`
    val nonEmptyBuffers = mutable.ArrayBuffer[DeviceMemoryBuffer]()
    toClose.iterator().asScala.foreach {
      case buf: DeviceMemoryBuffer if buf.getAddress > 0L && buf.getLength > 0L =>
        nonEmptyBuffers += buf
      case _: DeviceMemoryBuffer =>
      case _ =>
        throw new IllegalArgumentException(s"[$cv] ${toClose.asScala.mkString("|")} " +
          "broke the assumption all buffers are DeviceMemoryBuffer")
    }
    nonEmptyBuffers.toArray
  }

  // Super hacky help function which is only for moving the ownership of materialized column
  // from its owner ColumnVector wrapper to the top-level ColumnVector.
  // The top-level ColumnVector was created from Java side (HostToDevice), so its columnHandle
  // is unset. Meanwhile, the materialized column was created from native side, which means its
  // columnHandle was set. We can move the columnHandle of the materialized column to the target
  // ColumnVector. By doing that, the materialized column becomes a ColumnView, and the underlying
  // buffers will be cleaned up when we close the target top-level ColumnVector.
  private def moveColumnHandle(from: ColumnVector, to: ColumnVector): Unit = {
    val fromState = offHeapField.get(from)
    val toState = offHeapField.get(to)

    val colHandleField = fromState.getClass.getDeclaredField("columnHandle")
    colHandleField.setAccessible(true)

    val fromHandle = colHandleField.get(fromState).asInstanceOf[Long]
    colHandleField.setLong(fromState, 0L)
    colHandleField.setLong(toState, fromHandle)
  }

  private val offHeapField = {
    val field = classOf[ColumnView].getDeclaredField("offHeap")
    field.setAccessible(true)
    field
  }

  private val getDataAddr = {
    val method = classOf[ColumnView]
      .getDeclaredMethod("getNativeDataAddress", java.lang.Long.TYPE)
    method.setAccessible(true)
    method
  }
  private val getOffsetAddr = {
    val method = classOf[ColumnView]
      .getDeclaredMethod("getNativeOffsetsAddress", java.lang.Long.TYPE)
    method.setAccessible(true)
    method
  }
  private val getValidAddr = {
    val method = classOf[ColumnView]
      .getDeclaredMethod("getNativeValidityAddress", java.lang.Long.TYPE)
    method.setAccessible(true)
    method
  }
}
