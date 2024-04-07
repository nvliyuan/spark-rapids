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

package org.apache.spark.sql.execution.vectorized.rapids;

import ai.rapids.cudf.HostMemoryBuffer;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.unsafe.types.UTF8String;

import java.nio.ByteBuffer;

public class ParquetHelperVector extends WritableColumnVector {

	public enum PinMode {
		ENABLED,
		DISABLED,
		SYSTEM_DEFAULT
	}

	private HostMemoryBuffer data = null;

	private final PinMode pinMode;

	public ParquetHelperVector(int capacity, PinMode pinnedMode) {
		super(capacity, DataTypes.IntegerType);
		this.pinMode = pinnedMode;
		reserveInternal(capacity);
	}

	@Override
	protected void reserveInternal(int capacity) {
		resetBuffer(capacity, true);
	}

	public void resetBuffer(int capacity, boolean copyData) {
		long targetSize = capacity * 4L;
		long currentSize = data == null ? 0L : data.getLength();

		if (targetSize > currentSize) {
			HostMemoryBuffer buffer = allocateBuffer(targetSize);
			if (currentSize > 0) {
				if (copyData) {
					buffer.copyFromHostBuffer(0, data, 0, currentSize);
				}
				data.close();
			}
			data = buffer;
			this.capacity = capacity;
		}
	}

	private HostMemoryBuffer allocateBuffer(long size) {
		if (pinMode == PinMode.DISABLED) {
			return HostMemoryBuffer.allocate(size, false);
		}
		if (pinMode == PinMode.ENABLED) {
			return HostMemoryBuffer.allocate(size, true);
		}
		return HostMemoryBuffer.allocate(size);
	}

	public HostMemoryBuffer getRawBuffer() {
		return data;
	}

	@Override
	public int getDictId(int rowId) {
		return data.getInt(rowId * 4L);
	}

	@Override
	public int getInt(int rowId) {
		return data.getInt(rowId * 4L);
	}

	@Override
	public void putInt(int rowId, int value) {
		data.setInt(rowId * 4L, value);
	}

	@Override
	public void putInts(int rowId, int count, int value) {
		if (value == 0) {
			data.setMemory(rowId * 4L, count * 4L, (byte) 0);
			return;
		}
		for (long offset = rowId * 4L; offset < rowId * 4L + count * 4L; offset += 4) {
			data.setInt(offset, value);
		}
	}

	@Override
	public void putInts(int rowId, int count, int[] src, int srcIndex) {
		data.setInts(rowId * 4L, src, srcIndex, count);
	}

	@Override
	public void putInts(int rowId, int count, byte[] src, int srcIndex) {
		data.setBytes(rowId * 4L, src, srcIndex, count * 4L);
	}

	@Override
	public void putIntsLittleEndian(int rowId, int count, byte[] src, int srcIndex) {
		data.setBytes(rowId * 4L, src, srcIndex, count * 4L);
	}

	@Override
	public void close() {
		if (data != null) {
			while (data.getRefCount() > 0)
				data.close();
		}
	}

	@Override
	public void putNotNull(int rowId) {
		throw new UnsupportedOperationException("ParquetHelperVector does NOT support putters");
	}

	@Override
	public void putNull(int rowId) {
		throw new UnsupportedOperationException("ParquetHelperVector does NOT support putters");
	}

	@Override
	public void putNulls(int rowId, int count) {
		throw new UnsupportedOperationException("ParquetHelperVector does NOT support putters");
	}

	@Override
	public void putNotNulls(int rowId, int count) {
		throw new UnsupportedOperationException("ParquetHelperVector does NOT support putters");
	}

	@Override
	public void putBoolean(int rowId, boolean value) {
		throw new UnsupportedOperationException("ParquetHelperVector does NOT support putters");
	}

	@Override
	public void putBooleans(int rowId, int count, boolean value) {
		throw new UnsupportedOperationException("ParquetHelperVector does NOT support putters");
	}

	@Override
	public void putBooleans(int rowId, byte src) {
		throw new UnsupportedOperationException("ParquetHelperVector does NOT support putters");
	}

	@Override
	public void putByte(int rowId, byte value) {
		throw new UnsupportedOperationException("ParquetHelperVector does NOT support putters");
	}

	@Override
	public void putBytes(int rowId, int count, byte value) {
		throw new UnsupportedOperationException("ParquetHelperVector does NOT support putters");
	}

	@Override
	public void putBytes(int rowId, int count, byte[] src, int srcIndex) {
		throw new UnsupportedOperationException("ParquetHelperVector does NOT support putters");
	}

	@Override
	public void putShort(int rowId, short value) {
		throw new UnsupportedOperationException("ParquetHelperVector does NOT support putters");
	}

	@Override
	public void putShorts(int rowId, int count, short value) {
		throw new UnsupportedOperationException("ParquetHelperVector does NOT support putters");
	}

	@Override
	public void putShorts(int rowId, int count, short[] src, int srcIndex) {
		throw new UnsupportedOperationException("ParquetHelperVector does NOT support putters");
	}

	@Override
	public void putShorts(int rowId, int count, byte[] src, int srcIndex) {
		throw new UnsupportedOperationException("ParquetHelperVector does NOT support putters");
	}

	@Override
	public void putLong(int rowId, long value) {
		throw new UnsupportedOperationException("ParquetHelperVector does NOT support putters");
	}

	@Override
	public void putLongs(int rowId, int count, long value) {
		throw new UnsupportedOperationException("ParquetHelperVector does NOT support putters");
	}

	@Override
	public void putLongs(int rowId, int count, long[] src, int srcIndex) {
		throw new UnsupportedOperationException("ParquetHelperVector does NOT support putters");
	}

	@Override
	public void putLongs(int rowId, int count, byte[] src, int srcIndex) {
		throw new UnsupportedOperationException("ParquetHelperVector does NOT support putters");
	}

	@Override
	public void putLongsLittleEndian(int rowId, int count, byte[] src, int srcIndex) {
		throw new UnsupportedOperationException("ParquetHelperVector does NOT support putters");
	}

	@Override
	public void putFloat(int rowId, float value) {
		throw new UnsupportedOperationException("ParquetHelperVector does NOT support putters");
	}

	@Override
	public void putFloats(int rowId, int count, float value) {
		throw new UnsupportedOperationException("ParquetHelperVector does NOT support putters");
	}

	@Override
	public void putFloats(int rowId, int count, float[] src, int srcIndex) {
		throw new UnsupportedOperationException("ParquetHelperVector does NOT support putters");
	}

	@Override
	public void putFloats(int rowId, int count, byte[] src, int srcIndex) {
		throw new UnsupportedOperationException("ParquetHelperVector does NOT support putters");
	}

	@Override
	public void putFloatsLittleEndian(int rowId, int count, byte[] src, int srcIndex) {
		throw new UnsupportedOperationException("ParquetHelperVector does NOT support putters");
	}

	@Override
	public void putDouble(int rowId, double value) {
		throw new UnsupportedOperationException("ParquetHelperVector does NOT support putters");
	}

	@Override
	public void putDoubles(int rowId, int count, double value) {
		throw new UnsupportedOperationException("ParquetHelperVector does NOT support putters");
	}

	@Override
	public void putDoubles(int rowId, int count, double[] src, int srcIndex) {
		throw new UnsupportedOperationException("ParquetHelperVector does NOT support putters");
	}

	@Override
	public void putDoubles(int rowId, int count, byte[] src, int srcIndex) {
		throw new UnsupportedOperationException("ParquetHelperVector does NOT support putters");
	}

	@Override
	public void putDoublesLittleEndian(int rowId, int count, byte[] src, int srcIndex) {
		throw new UnsupportedOperationException("ParquetHelperVector does NOT support putters");
	}

	@Override
	public void putArray(int rowId, int offset, int length) {
		throw new UnsupportedOperationException("ParquetHelperVector does NOT support putters");
	}

	@Override
	public int putByteArray(int rowId, byte[] value, int offset, int count) {
		throw new UnsupportedOperationException("ParquetHelperVector does NOT support putters");
	}

	@Override
	protected UTF8String getBytesAsUTF8String(int rowId, int count) {
		throw new UnsupportedOperationException("ParquetHelperVector does NOT support getters");
	}

	@Override
	public ByteBuffer getByteBuffer(int rowId, int count) {
		throw new UnsupportedOperationException("ParquetHelperVector does NOT support getters");
	}

	@Override
	public int getArrayLength(int rowId) {
		throw new UnsupportedOperationException("ParquetHelperVector does NOT support getters");
	}

	@Override
	public int getArrayOffset(int rowId) {
		throw new UnsupportedOperationException("ParquetHelperVector does NOT support getters");
	}

	@Override
	protected WritableColumnVector reserveNewColumn(int capacity, DataType type) {
		throw new UnsupportedOperationException("ParquetHelperVector does NOT support reserveNewColumn");
	}

	@Override
	public boolean isNullAt(int rowId) {
		throw new UnsupportedOperationException("ParquetHelperVector does NOT support getters");
	}

	@Override
	public boolean getBoolean(int rowId) {
		throw new UnsupportedOperationException("ParquetHelperVector does NOT support getters");
	}

	@Override
	public byte getByte(int rowId) {
		throw new UnsupportedOperationException("ParquetHelperVector does NOT support getters");
	}

	@Override
	public short getShort(int rowId) {
		throw new UnsupportedOperationException("ParquetHelperVector does NOT support getters");
	}

	@Override
	public long getLong(int rowId) {
		throw new UnsupportedOperationException("ParquetHelperVector does NOT support getters");
	}

	@Override
	public float getFloat(int rowId) {
		throw new UnsupportedOperationException("ParquetHelperVector does NOT support getters");
	}

	@Override
	public double getDouble(int rowId) {
		throw new UnsupportedOperationException("ParquetHelperVector does NOT support getters");
	}
}
