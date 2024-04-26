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

import java.io.IOException;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.parquet.bytes.ByteBufferInputStream;
import org.apache.parquet.column.values.ValuesReader;
import org.apache.parquet.io.api.Binary;
import org.apache.parquet.io.ParquetDecodingException;

import org.apache.spark.sql.execution.vectorized.rapids.WritableColumnVector;

/**
 * An implementation of the Parquet PLAIN decoder that supports the vectorized interface.
 */
public class VectorizedPlainValuesReader extends ValuesReader implements VectorizedValuesReader {

	private ByteBufferIsConsumer consumer = null;

	// Only used for booleans.
	private int bitOffset;
	private byte currentByte = 0;

	public VectorizedPlainValuesReader() {
	}

	@Override
	public void initFromPage(int valueCount, ByteBufferInputStream in) {
		consumer = ByteBufferIsConsumer.create(in);
	}

	@Override
	public void skip() {
		throw new UnsupportedOperationException();
	}

	private void updateCurrentByte() {
		currentByte = consumer.getByte();
	}

	@Override
	public final void readBooleans(int total, WritableColumnVector c, int rowId) {
		int i = 0;
		if (bitOffset > 0) {
			i = Math.min(8 - bitOffset, total);
			c.putBooleans(rowId, i, currentByte, bitOffset);
			bitOffset = (bitOffset + i) & 7;
		}
		for (; i + 7 < total; i += 8) {
			updateCurrentByte();
			c.putBooleans(rowId + i, currentByte);
		}
		if (i < total) {
			updateCurrentByte();
			bitOffset = total - i;
			c.putBooleans(rowId + i, bitOffset, currentByte, 0);
		}
	}

	@Override
	public final void skipBooleans(int total) {
		int i = 0;
		if (bitOffset > 0) {
			i = Math.min(8 - bitOffset, total);
			bitOffset = (bitOffset + i) & 7;
		}
		if (i + 7 < total) {
			int numBytesToSkip = (total - i) / 8;
			try {
				consumer.advance(numBytesToSkip);
			} catch (Exception e) {
				throw new ParquetDecodingException("Failed to skip bytes", e);
			}
			i += numBytesToSkip * 8;
		}
		if (i < total) {
			updateCurrentByte();
			bitOffset = total - i;
		}
	}

	@Override
	public final void readIntegers(int total, WritableColumnVector c, int rowId) {
		consumer.readInts(total, c, rowId);
	}

	@Override
	public void skipIntegers(int total) {
		consumer.advance(total * 4L);
	}

	@Override
	public final void readUnsignedIntegers(int total, WritableColumnVector c, int rowId) {
		consumer.readUIntsAsLongs(total, c, rowId);
	}

	@Override
	public final void readIntegersWithRebase(
			int total, WritableColumnVector c, int rowId, boolean failIfRebase) {
		throw new NotImplementedException("Due to performance issues, readIntegersWithRebase is not supported");
	}

	@Override
	public final void readLongs(int total, WritableColumnVector c, int rowId) {
		consumer.readLongs(total, c, rowId);
	}

	@Override
	public void skipLongs(int total) {
		consumer.advance(total * 8L);
	}

	@Override
	public final void readUnsignedLongs(int total, WritableColumnVector c, int rowId) {
		throw new NotImplementedException("Due to performance issues, readUnsignedLongs is not supported");
	}

	@Override
	public final void readLongsWithRebase(
			int total,
			WritableColumnVector c,
			int rowId,
			boolean failIfRebase,
			String timeZone) {
		throw new NotImplementedException("Due to performance issues, readLongsWithRebase is not supported");
	}

	@Override
	public final void readFloats(int total, WritableColumnVector c, int rowId) {
		consumer.readFloats(total, c, rowId);
	}

	@Override
	public void skipFloats(int total) {
		consumer.advance(total * 4L);
	}

	@Override
	public final void readDoubles(int total, WritableColumnVector c, int rowId) {
		consumer.readDoubles(total, c, rowId);
	}

	@Override
	public void skipDoubles(int total) {
		consumer.advance(total * 8L);
	}

	@Override
	public final void readBytes(int total, WritableColumnVector c, int rowId) {
		// Bytes are stored as a 4-byte little endian int. Just read the first byte.
		consumer.readIntsAsBytes(total, c, rowId);
	}

	@Override
	public final void skipBytes(int total) {
		consumer.advance(total * 4L);
	}

	@Override
	public final void readShorts(int total, WritableColumnVector c, int rowId) {
		consumer.readIntsAsShorts(total, c, rowId);
	}

	@Override
	public void skipShorts(int total) {
		consumer.advance(total * 2L);
	}

	@Override
	public final boolean readBoolean() {
		if (bitOffset == 0) {
			updateCurrentByte();
		}

		boolean v = (currentByte & (1 << bitOffset)) != 0;
		bitOffset += 1;
		if (bitOffset == 8) {
			bitOffset = 0;
		}
		return v;
	}

	@Override
	public final int readInteger() {
		return consumer.getInt();
	}

	@Override
	public final long readLong() {
		return consumer.getLong();
	}

	@Override
	public final byte readByte() {
		return (byte) consumer.getInt();
	}

	@Override
	public short readShort() {
		return (short) consumer.getInt();
	}

	@Override
	public final float readFloat() {
		return consumer.getFloat();
	}

	@Override
	public final double readDouble() {
		return consumer.getDouble();
	}

	@Override
	public final void readBinary(int total, WritableColumnVector v, int rowId) {
		consumer.readBinaries(total, v, rowId);
	}

	@Override
	public void skipBinary(int total) {
		for (int i = 0; i < total; i++) {
			int len = readInteger();
			consumer.advance(len);
		}
	}

	@Override
	public final Binary readBinary(int len) {
		return consumer.getBinary(len);
	}

	@Override
	public void skipFixedLenByteArray(int total, int len) {
		consumer.advance(total * (long) len);
	}
}
