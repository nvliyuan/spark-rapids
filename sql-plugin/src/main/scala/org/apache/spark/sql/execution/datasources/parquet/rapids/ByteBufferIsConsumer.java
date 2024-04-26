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
import org.apache.spark.sql.execution.vectorized.rapids.WritableColumnVector;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Iterator;
import java.util.List;

public abstract class ByteBufferIsConsumer {

	protected Iterator<ByteBuffer> iterator;
	protected ByteBuffer current = null;

	public ByteBufferIsConsumer(Iterator<ByteBuffer> bufferIterator) {
		iterator = bufferIterator;
		if (iterator.hasNext()) {
			pointToNextBuffer();
		}
	}

	protected void pointToNextBuffer() {
		current = iterator.next();
		current.order(ByteOrder.LITTLE_ENDIAN);
	}

	public void advance(long sizeInByte) {
		long remaining = sizeInByte;
		while (remaining > 0) {
			if (!current.hasRemaining()) pointToNextBuffer();

			int batchSize = (remaining >= current.remaining()) ? current.remaining() : (int) remaining;
			current.position(current.position() + batchSize);
			remaining -= batchSize;
		}
	}

	public abstract void readInts(int total, WritableColumnVector c, int rowId);
	public abstract void readLongs(int total, WritableColumnVector c, int rowId);
	public abstract void readFloats(int total, WritableColumnVector c, int rowId);
	public abstract void readDoubles(int total, WritableColumnVector c, int rowId);
	public abstract void readUIntsAsLongs(int total, WritableColumnVector c, int rowId);
	public abstract void readIntsAsShorts(int total, WritableColumnVector c, int rowId);
	public abstract void readIntsAsBytes(int total, WritableColumnVector c, int rowId);
	public abstract void readBinaries(int total, WritableColumnVector c, int rowId);
	public abstract byte getByte();
	public abstract int getInt();
	public abstract long getLong();
	public abstract float getFloat();
	public abstract double getDouble();
	public abstract Binary getBinary(int len);

	public static ByteBufferIsConsumer create(ByteBufferInputStream bis) {
		List<ByteBuffer> buffers = bis.remainingBuffers();
		if (buffers.isEmpty()) {
			throw new IllegalArgumentException("Got empty ByteBufferInputStream");
		}
		if (buffers.size() > 1) {
			System.err.printf("create a MultiByteBuffersConsumer with %d buffers\n", buffers.size());
		}
		// HeapByteBufferIsConsumer for HeapByteBuffer; DirectByteBufferIsConsumer for DirectByteBuffer
		if (buffers.get(0).hasArray()) {
			return new HeapByteBufferIsConsumer(buffers.iterator());
		}
		return new DirectByteBufferIsConsumer(buffers.iterator());
	}

}
