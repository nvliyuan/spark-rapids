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

package org.apache.spark.sql.execution.datasources.parquet.rapids;

import java.io.Closeable;
import java.io.IOException;
import java.time.ZoneId;

import org.apache.parquet.CorruptDeltaByteArrays;
import org.apache.parquet.VersionParser.ParsedVersion;
import org.apache.parquet.bytes.ByteBufferInputStream;
import org.apache.parquet.bytes.BytesInput;
import org.apache.parquet.bytes.BytesUtils;
import org.apache.parquet.column.ColumnDescriptor;
import org.apache.parquet.column.Dictionary;
import org.apache.parquet.column.Encoding;
import org.apache.parquet.column.page.*;
import org.apache.parquet.column.values.RequiresPreviousReader;
import org.apache.parquet.column.values.ValuesReader;
import org.apache.parquet.column.values.dictionary.PlainValuesDictionary.PlainBinaryDictionary;
import org.apache.parquet.schema.LogicalTypeAnnotation;
import org.apache.parquet.schema.PrimitiveType;

import org.apache.spark.sql.execution.vectorized.rapids.WritableColumnVector;
import org.apache.spark.sql.types.DataTypes;

import static org.apache.parquet.schema.PrimitiveType.PrimitiveTypeName.BOOLEAN;

/**
 * Decoder to return values from a single column.
 */
public class VectorizedColumnReader implements Closeable {
  /**
   * The dictionary, if this column has dictionary encoding.
   */
  private Dictionary dictionary;

  /**
   * If true, the current page is dictionary encoded.
   */
  private boolean isCurrentPageDictionaryEncoded;

  /**
   * Value readers.
   */
  private ValuesReader dataColumn;

  /**
   * Vectorized RLE decoder for definition levels
   */
  private VectorizedRleValuesReader defColumn;

  /**
   * Vectorized RLE decoder for repetition levels
   */
  private VectorizedRleValuesReader repColumn;

  /**
   * Factory to get type-specific vector updater.
   */
  private final ParquetVectorUpdaterFactory updaterFactory;

  /**
   * Helper struct to track intermediate states while reading Parquet pages in the column chunk.
   */
  private final ParquetReadState readState;

  /**
   * The index for the first row in the current page, among all rows across all pages in the
   * column chunk for this reader. If there is no column index, the value is 0.
   */
  private long pageFirstRowIndex;

  private final PageReader pageReader;
  private final ColumnDescriptor descriptor;
  private final LogicalTypeAnnotation logicalTypeAnnotation;
  private final String datetimeRebaseMode;
  private final ParsedVersion writerVersion;

  private final boolean dictLateMaterialize;

  private int pageIndex = 0;

  public VectorizedColumnReader(
      ColumnDescriptor descriptor,
      boolean isRequired,
      int maxRepetitiveDefLevel,
      PageReadStore pageReadStore,
      ZoneId convertTz,
      String datetimeRebaseMode,
      String datetimeRebaseTz,
      String int96RebaseMode,
      String int96RebaseTz,
      ParsedVersion writerVersion,
      boolean supportDictLateMaterialize) throws IOException {
    this.descriptor = descriptor;
    this.pageReader = pageReadStore.getPageReader(descriptor);
    this.readState = new ParquetReadState(descriptor, isRequired,
      pageReadStore.getRowIndexes().orElse(null));
    this.readState.maxRepetitiveDefLevel = maxRepetitiveDefLevel;
    this.logicalTypeAnnotation = descriptor.getPrimitiveType().getLogicalTypeAnnotation();
    this.updaterFactory = new ParquetVectorUpdaterFactory(
      logicalTypeAnnotation,
      convertTz,
      datetimeRebaseMode,
      datetimeRebaseTz,
      int96RebaseMode,
      int96RebaseTz);
    this.dictLateMaterialize = supportDictLateMaterialize;

    if (!this.dictLateMaterialize) {
      DictionaryPage dictionaryPage = pageReader.readDictionaryPage();
      if (dictionaryPage != null) {
        try {
          this.dictionary = dictionaryPage.getEncoding().initDictionary(descriptor, dictionaryPage);
          this.isCurrentPageDictionaryEncoded = true;
        } catch (IOException e) {
          throw new IOException("could not decode the dictionary for " + descriptor, e);
        }
      } else {
        this.dictionary = null;
        this.isCurrentPageDictionaryEncoded = false;
      }
    }

    if (pageReader.getTotalValueCount() == 0) {
      throw new IOException("totalValueCount == 0");
    }
    assert "LEGACY".equals(datetimeRebaseMode) || "EXCEPTION".equals(datetimeRebaseMode) ||
      "CORRECTED".equals(datetimeRebaseMode);
    this.datetimeRebaseMode = datetimeRebaseMode;
    assert "LEGACY".equals(int96RebaseMode) || "EXCEPTION".equals(int96RebaseMode) ||
      "CORRECTED".equals(int96RebaseMode);
    this.writerVersion = writerVersion;
  }

  public Dictionary getDictionary() {
    return dictionary;
  }

  /**
   * Reads `total` rows from this columnReader into column.
   */
  void readBatch(
      int total,
      WritableColumnVector column,
      WritableColumnVector repetitionLevels,
      WritableColumnVector definitionLevels) throws IOException {

    ParquetVectorUpdater updater;
    if (!dictLateMaterialize) {
      updater = updaterFactory.getUpdater(descriptor, column.dataType());
    } else {
      updater = new ParquetVectorUpdaterFactory.IntegerUpdater();
    }

    WritableColumnVector dictionaryIds = null;

    if (dictLateMaterialize) {
      column.reserveAdditional(total);
    } else if (dictionary != null) {
      // SPARK-16334: We only maintain a single dictionary per row batch, so that it can be used to
      // decode all previous dictionary encoded pages if we ever encounter a non-dictionary encoded
      // page.
      dictionaryIds = column.reserveDictionaryIds(total);

      // When decoding String from BinaryDictionary, we can leverage ZerocopyStringUpdater to get rid of
      // the overhead of Java wrappers of on heap data.
      if (dictionary instanceof PlainBinaryDictionary &&
          column.dataType() == DataTypes.StringType) {
        dictionary = new OffHeapBinaryDictionary((PlainBinaryDictionary) dictionary);
      }
    }

    readState.resetForNewBatch(total);
    while (readState.rowsToReadInBatch > 0 || !readState.lastListCompleted) {
      if (readState.valuesToReadInPage == 0) {
        int pageValueCount = readPage();
        pageIndex++;
        if (pageValueCount < 0) {
          // we've read all the pages; this could happen when we're reading a repeated list and we
          // don't know where the list will end until we've seen all the pages.
          break;
        }
        readState.resetForNewPage(pageValueCount, pageFirstRowIndex);
      }

      if (!dictLateMaterialize && isCurrentPageDictionaryEncoded) {
        // Save starting offset in case we need to decode dictionary IDs.
        int startOffset = readState.valueOffset;

        // Read and decode dictionary ids.
        if (readState.maxRepetitionLevel == 0) {
          defColumn.readIntegers(readState, dictionaryIds, column, definitionLevels,
            (VectorizedValuesReader) dataColumn);
        } else {
          repColumn.readIntegersRepeated(readState, repetitionLevels, defColumn, definitionLevels,
            dictionaryIds, column, (VectorizedValuesReader) dataColumn);
        }

				updater.decodeDictionaryIds(readState.valueOffset - startOffset, startOffset, column,
					dictionaryIds, dictionary);
      } else {

        VectorizedValuesReader valuesReader = (VectorizedValuesReader) dataColumn;
        if (readState.maxRepetitionLevel == 0) {
          defColumn.readBatch(readState, column, definitionLevels, valuesReader, updater);
        } else {
          repColumn.readBatchRepeated(readState, repetitionLevels, defColumn, definitionLevels,
            column, valuesReader, updater);
        }
      }
    }
  }

  private int readPage() {
    DataPage page = pageReader.readPage();
    if (page == null) {
      return -1;
    }
    this.pageFirstRowIndex = page.getFirstRowIndex().orElse(0L);

    return page.accept(new DataPage.Visitor<Integer>() {
      @Override
      public Integer visit(DataPageV1 dataPageV1) {
        try {
          return readPageV1(dataPageV1);
        } catch (IOException e) {
          throw new RuntimeException(e);
        }
      }

      @Override
      public Integer visit(DataPageV2 dataPageV2) {
        try {
          return readPageV2(dataPageV2);
        } catch (IOException e) {
          throw new RuntimeException(e);
        }
      }
    });
  }

  private void initDataReader(
      int pageValueCount,
      Encoding dataEncoding,
      ByteBufferInputStream in) throws IOException {
    ValuesReader previousReader = this.dataColumn;
    if (dataEncoding.usesDictionary()) {
      this.dataColumn = null;
      if (!dictLateMaterialize && dictionary == null) {
        throw new IOException(
            "could not read page in col " + descriptor +
                " as the dictionary was missing for encoding " + dataEncoding);
      }
      @SuppressWarnings("deprecation")
      Encoding plainDict = Encoding.PLAIN_DICTIONARY; // var to allow warning suppression
      if (dataEncoding != plainDict && dataEncoding != Encoding.RLE_DICTIONARY) {
        throw new UnsupportedOperationException("Unsupported encoding: " + dataEncoding);
      }
      this.dataColumn = new VectorizedRleValuesReader();
      this.isCurrentPageDictionaryEncoded = true;
    } else {
      if (dictLateMaterialize) {
        StringBuilder sb = new StringBuilder();
        for (String p : descriptor.getPath()) {
          sb.append(p).append('.');
        }
        throw new RuntimeException("DictLatMat failed: Column(" + sb + ") Page_NO(" +
            pageIndex + ") Encoding(" + dataEncoding + ") is not dictionary encoded");
      }
      this.dataColumn = getValuesReader(dataEncoding);
      this.isCurrentPageDictionaryEncoded = false;
    }

    try {
      dataColumn.initFromPage(pageValueCount, in);
    } catch (IOException e) {
      throw new IOException("could not read page in col " + descriptor, e);
    }
    // for PARQUET-246 (See VectorizedDeltaByteArrayReader.setPreviousValues)
    if (CorruptDeltaByteArrays.requiresSequentialReads(writerVersion, dataEncoding) &&
        previousReader instanceof RequiresPreviousReader) {
      // previousReader can only be set if reading sequentially
      ((RequiresPreviousReader) dataColumn).setPreviousReader(previousReader);
    }
  }

  private ValuesReader getValuesReader(Encoding encoding) {
    switch (encoding) {
      case PLAIN:
        return new VectorizedPlainValuesReader();
      case DELTA_BYTE_ARRAY:
        return new VectorizedDeltaByteArrayReader();
      case DELTA_LENGTH_BYTE_ARRAY:
        return new VectorizedDeltaLengthByteArrayReader();
      case DELTA_BINARY_PACKED:
        return new VectorizedDeltaBinaryPackedReader();
      case RLE:
        PrimitiveType.PrimitiveTypeName typeName =
          this.descriptor.getPrimitiveType().getPrimitiveTypeName();
        // RLE encoding only supports boolean type `Values`, and  `bitwidth` is always 1.
        if (typeName == BOOLEAN) {
          return new VectorizedRleValuesReader(1);
        } else {
          throw new UnsupportedOperationException(
            "RLE encoding is not supported for values of type: " + typeName);
        }
      default:
        throw new UnsupportedOperationException("Unsupported encoding: " + encoding);
    }
  }


  private int readPageV1(DataPageV1 page) throws IOException {
    if (page.getDlEncoding() != Encoding.RLE && descriptor.getMaxDefinitionLevel() != 0) {
      throw new UnsupportedOperationException("Unsupported encoding: " + page.getDlEncoding());
    }

    int pageValueCount = page.getValueCount();

    int rlBitWidth = BytesUtils.getWidthFromMaxInt(descriptor.getMaxRepetitionLevel());
    this.repColumn = new VectorizedRleValuesReader(rlBitWidth);

    int dlBitWidth = BytesUtils.getWidthFromMaxInt(descriptor.getMaxDefinitionLevel());
    this.defColumn = new VectorizedRleValuesReader(dlBitWidth);

    try {
      BytesInput bytes = page.getBytes();
      ByteBufferInputStream in = bytes.toInputStream();

      repColumn.initFromPage(pageValueCount, in);
      defColumn.initFromPage(pageValueCount, in);
      initDataReader(pageValueCount, page.getValueEncoding(), in);
      return pageValueCount;
    } catch (IOException e) {
      throw new IOException("could not read page " + page + " in col " + descriptor, e);
    }
  }

  private int readPageV2(DataPageV2 page) throws IOException {
    int pageValueCount = page.getValueCount();

    // do not read the length from the stream. v2 pages handle dividing the page bytes.
    int rlBitWidth = BytesUtils.getWidthFromMaxInt(descriptor.getMaxRepetitionLevel());
    repColumn = new VectorizedRleValuesReader(rlBitWidth, false);
    repColumn.initFromPage(pageValueCount, page.getRepetitionLevels().toInputStream());

    int dlBitWidth = BytesUtils.getWidthFromMaxInt(descriptor.getMaxDefinitionLevel());
    defColumn = new VectorizedRleValuesReader(dlBitWidth, false);
    defColumn.initFromPage(pageValueCount, page.getDefinitionLevels().toInputStream());

    try {
      initDataReader(pageValueCount, page.getDataEncoding(), page.getData().toInputStream());
      return pageValueCount;
    } catch (IOException e) {
      throw new IOException("could not read page " + page + " in col " + descriptor, e);
    }
  }

  @Override
  public void close() {
    if (dictionary != null && dictionary instanceof OffHeapBinaryDictionary) {
      ((OffHeapBinaryDictionary) dictionary).close();
    }
  }
}
