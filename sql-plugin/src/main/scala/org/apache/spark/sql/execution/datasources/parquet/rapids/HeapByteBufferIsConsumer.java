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

import org.apache.parquet.bytes.ByteBufferInputStream;
import org.apache.parquet.io.api.Binary;
import org.apache.spark.sql.execution.vectorized.rapids.RapidsWritableColumnVector;
import org.apache.spark.sql.execution.vectorized.rapids.WritableColumnVector;

import java.nio.ByteBuffer;
import java.util.Iterator;

public class HeapByteBufferIsConsumer extends ByteBufferIsConsumer {

	// Reference of underlying data structure of HeapByteBuffer, since we assume the BIS is
	// backed by the HeapByteBuffer
	byte[] hb;
	int arrayOffset;

	public HeapByteBufferIsConsumer(Iterator<ByteBuffer> bufferIterator) {
		super(bufferIterator);
	}

	protected void pointToNextBuffer() {
		super.pointToNextBuffer();
		assert current.hasArray();
		hb = current.array();
		arrayOffset = current.arrayOffset();
	}

	public void readInts(int total, WritableColumnVector c, int rowId) {
		int remaining = total * 4;
		int tgtOffset = rowId;

		while (remaining > 0) {
			if (!current.hasRemaining()) pointToNextBuffer();

			int size = Math.min(remaining, current.remaining());
			// TODO: handle buffer tails separately, when there are multiple buffers
			if (size >> 2 << 2 != size) {
				throw new RuntimeException("Will support the special handling of buffer tails, when there are multiple buffers");
			}
			int srcOffset = this.arrayOffset + current.position();
			int sizeInRow = size >> 2;
			c.putIntsLittleEndian(tgtOffset, sizeInRow, hb, srcOffset);
			current.position(current.position() + size);
			tgtOffset += sizeInRow;
			remaining -= size;
		}
	}

	public void readLongs(int total, WritableColumnVector c, int rowId) {
		int remaining = total * 8;
		int tgtOffset = rowId;

		while (remaining > 0) {
			if (!current.hasRemaining()) pointToNextBuffer();

			int size = Math.min(remaining, current.remaining());
			// TODO: handle buffer tails separately, when there are multiple buffers
			if (size >> 3 << 3 != size) {
				throw new RuntimeException("Will support the special handling of buffer tails, when there are multiple buffers");
			}
			int srcOffset = this.arrayOffset + current.position();
			int sizeInRow = size >> 3;
			c.putLongsLittleEndian(tgtOffset, sizeInRow, hb, srcOffset);
			current.position(current.position() + size);
			tgtOffset += sizeInRow;
			remaining -= size;
		}
	}

	public void readFloats(int total, WritableColumnVector c, int rowId) {
		int remaining = total * 4;
		int tgtOffset = rowId;

		while (remaining > 0) {
			if (!current.hasRemaining()) pointToNextBuffer();

			int size = Math.min(remaining, current.remaining());
			// TODO: handle buffer tails separately, when there are multiple buffers
			if (size >> 2 << 2 != size) {
				throw new RuntimeException("Will support the special handling of buffer tails, when there are multiple buffers");
			}
			int srcOffset = this.arrayOffset + current.position();
			int sizeInRow = size >> 2;
			c.putFloatsLittleEndian(tgtOffset, sizeInRow, hb, srcOffset);
			current.position(current.position() + size);
			tgtOffset += sizeInRow;
			remaining -= size;
		}
	}

	public void readDoubles(int total, WritableColumnVector c, int rowId) {
		int remaining = total * 8;
		int tgtOffset = rowId;

		while (remaining > 0) {
			if (!current.hasRemaining()) pointToNextBuffer();

			int size = Math.min(remaining, current.remaining());
			// TODO: handle buffer tails separately, when there are multiple buffers
			if (size >> 3 << 3 != size) {
				throw new RuntimeException("Will support the special handling of buffer tails, when there are multiple buffers");
			}
			int srcOffset = this.arrayOffset + current.position();
			int sizeInRow = size >> 3;
			c.putDoublesLittleEndian(tgtOffset, sizeInRow, hb, srcOffset);
			current.position(current.position() + size);
			tgtOffset += sizeInRow;
			remaining -= size;
		}
	}

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

	public void readBinaries(int total, WritableColumnVector c, int rowId) {
		RapidsWritableColumnVector vector = (RapidsWritableColumnVector) c;
		WritableColumnVector charVector = c.arrayData();

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
					charVector.putBytes(charOffset, size, this.hb, this.arrayOffset + bufPos);
					charOffset += size;
					current.position(bufPos + size);
					remainLen -= size;
				}
				charVector.addElementsAppended(curLength);
			}

			vector.commitStringAppend(rowId + i, prevOffset, curLength);
		}
	}

	public byte getByte() {
		if (!current.hasRemaining()) pointToNextBuffer();
		return current.get();
	}

	public int getInt() {
		if (!current.hasRemaining()) pointToNextBuffer();
		return current.getInt();
	}

	public long getLong() {
		if (!current.hasRemaining()) pointToNextBuffer();
		return current.getLong();
	}

	public float getFloat() {
		if (!current.hasRemaining()) pointToNextBuffer();
		return current.getFloat();
	}

	public double getDouble() {
		if (!current.hasRemaining()) pointToNextBuffer();
		return current.getDouble();
	}

	public Binary getBinary(int len) {
		byte[] target = new byte[len];
		int targetOffset = 0;

		do {
			if (!current.hasRemaining()) pointToNextBuffer();

			int batchSize = Math.min(len - targetOffset, current.remaining());
			System.arraycopy(hb,this.arrayOffset + current.position(),
					target, targetOffset, batchSize);
			current.position(current.position() + batchSize);
			targetOffset += batchSize;
		}
		while (targetOffset < len);

		return Binary.fromConstantByteArray(target);
	}

}
