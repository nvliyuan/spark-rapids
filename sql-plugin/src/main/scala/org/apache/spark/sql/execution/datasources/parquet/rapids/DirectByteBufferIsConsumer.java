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

import java.lang.reflect.Field;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.util.Iterator;

import org.apache.parquet.bytes.ByteBufferInputStream;
import org.apache.parquet.io.api.Binary;
import org.apache.spark.sql.execution.vectorized.rapids.RapidsWritableColumnVector;
import org.apache.spark.sql.execution.vectorized.rapids.UnsafeMemoryUtils;
import org.apache.spark.sql.execution.vectorized.rapids.WritableColumnVector;

public class DirectByteBufferIsConsumer extends ByteBufferIsConsumer {

	static Field addrField;

	static {
		try {
			addrField = Buffer.class.getDeclaredField("address");
			addrField.setAccessible(true);
		} catch (NoSuchFieldException e) {
			throw new RuntimeException(e);
		}
	}

	private long address;

	public DirectByteBufferIsConsumer(Iterator<ByteBuffer> bufferIterator) {
		super(bufferIterator);
	}

	protected void pointToNextBuffer() {
		super.pointToNextBuffer();
		assert current.isDirect();
		try {
			address = addrField.getLong(current);
		} catch (IllegalAccessException e) {
			throw new RuntimeException(e);
		}
	}

	private void readFixedLengthData(int total, WritableColumnVector c, int rowId, int bitWidth) {
		assert c instanceof RapidsWritableColumnVector : "Only supports RapidsWritableColumnVector";
		RapidsWritableColumnVector cv = (RapidsWritableColumnVector) c;

		int remaining = total << bitWidth;
		int tgtOffset = rowId;

		while (remaining > 0) {
			if (!current.hasRemaining()) pointToNextBuffer();

			int size = Math.min(remaining, current.remaining());
			// TODO: handle buffer tails separately, when there are multiple buffers
			if (size >> 2 << 2 != size) {
				throw new RuntimeException("Will support the special handling of buffer tails, when there are multiple buffers");
			}
			long srcOffset = address + current.position();
			int sizeInRow = size >> bitWidth;
			cv.putFixedLengthElementsUnsafely(tgtOffset, srcOffset, size, bitWidth);
			current.position(current.position() + size);
			tgtOffset += sizeInRow;
			remaining -= size;
		}
	}

	@Override
	public void readInts(int total, WritableColumnVector c, int rowId) {
		readFixedLengthData(total, c, rowId, 2);
	}

	@Override
	public void readLongs(int total, WritableColumnVector c, int rowId) {
		readFixedLengthData(total, c, rowId, 3);
	}

	@Override
	public void readFloats(int total, WritableColumnVector c, int rowId) {
		readFixedLengthData(total, c, rowId, 2);
	}

	@Override
	public void readDoubles(int total, WritableColumnVector c, int rowId) {
		readFixedLengthData(total, c, rowId, 3);
	}

	@Override
	public void readBinaries(int total, WritableColumnVector c, int rowId) {
		assert c instanceof RapidsWritableColumnVector : "Only supports RapidsWritableColumnVector";
		RapidsWritableColumnVector cv = (RapidsWritableColumnVector) c;
		RapidsWritableColumnVector charVector = (RapidsWritableColumnVector) cv.arrayData();

		for (int i = 0; i < total; ++i) {
			if (!current.hasRemaining()) pointToNextBuffer();

			int curLength = current.getInt();
			int prevOffset = charVector.getElementsAppended();

			if (curLength > 0) {
				charVector.reserve(prevOffset + curLength);

				int remainLen = curLength;
				int charOffset = prevOffset;
				while (remainLen > 0) {
					if (!current.hasRemaining()) pointToNextBuffer();

					int size = Math.min(remainLen, current.remaining());
					int bufPos = current.position();
					charVector.putFixedLengthElementsUnsafely(charOffset, this.address + bufPos, size, 0);
					charOffset += size;
					current.position(bufPos + size);
					remainLen -= size;
				}
				charVector.addElementsAppended(curLength);
			}

			cv.commitStringAppend(rowId + i, prevOffset, curLength);
		}
	}

	@Override
	public void readUIntsAsLongs(int total, WritableColumnVector c, int rowId) {
		int remaining = total * 4;
		int tgtOffset = rowId;

		while (remaining > 0) {
			if (!current.hasRemaining()) pointToNextBuffer();

			int size = Math.min(remaining, current.remaining());
			for (int i = 0; i < size >> 2; ++i) {
				c.putLong(tgtOffset + i, Integer.toUnsignedLong(current.getInt()));
			}
			tgtOffset += size >> 3;
			remaining -= size;
		}
	}

	@Override
	public void readIntsAsShorts(int total, WritableColumnVector c, int rowId) {
		int remaining = total * 4;
		int tgtOffset = rowId;

		while (remaining > 0) {
			if (!current.hasRemaining()) pointToNextBuffer();

			int size = Math.min(remaining, current.remaining());
			for (int i = 0; i < size >> 2; ++i) {
				c.putShort(tgtOffset + i, (short) current.getInt());
			}
			tgtOffset += size >> 1;
			remaining -= size;
		}
	}

	@Override
	public void readIntsAsBytes(int total, WritableColumnVector c, int rowId) {
		int remaining = total * 4;
		int tgtOffset = rowId;

		while (remaining > 0) {
			if (!current.hasRemaining()) pointToNextBuffer();

			int size = Math.min(remaining, current.remaining());
			int pos = current.position();
			for (int i = 0; i < size >> 2; ++i) {
				c.putByte(tgtOffset + i, current.get(pos + (i << 2)));
			}
			tgtOffset += size;
			current.position(pos + size);
			remaining -= size;
		}
	}

	@Override
	public byte getByte() {
		if (!current.hasRemaining()) pointToNextBuffer();
		return current.get();
	}

	@Override
	public int getInt() {
		if (!current.hasRemaining()) pointToNextBuffer();
		return current.getInt();
	}

	@Override
	public long getLong() {
		if (!current.hasRemaining()) pointToNextBuffer();
		return current.getLong();
	}

	@Override
	public float getFloat() {
		if (!current.hasRemaining()) pointToNextBuffer();
		return current.getFloat();
	}

	@Override
	public double getDouble() {
		if (!current.hasRemaining()) pointToNextBuffer();
		return current.getDouble();
	}

	@Override
	public Binary getBinary(int len) {
		byte[] target = new byte[len];
		int targetOffset = 0;

		do {
			if (!current.hasRemaining()) pointToNextBuffer();

			int batchSize = Math.min(len - targetOffset, current.remaining());
			UnsafeMemoryUtils.copyMemory(null,
					this.address + current.position(), target, targetOffset, batchSize);
			current.position(current.position() + batchSize);
			targetOffset += batchSize;
		}
		while (targetOffset < len);

		return Binary.fromConstantByteArray(target);
	}

}
