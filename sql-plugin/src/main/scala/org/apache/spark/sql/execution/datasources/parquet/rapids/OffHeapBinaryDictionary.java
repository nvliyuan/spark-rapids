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

import ai.rapids.cudf.HostMemoryBuffer;
import org.apache.parquet.column.Dictionary;
import org.apache.parquet.column.values.dictionary.PlainValuesDictionary.PlainBinaryDictionary;

public class OffHeapBinaryDictionary extends Dictionary implements Closeable {

	public OffHeapBinaryDictionary(PlainBinaryDictionary binDict) {
		super(binDict.getEncoding());
		this.size = binDict.getMaxId() + 1;
		offsets = new int[this.size + 1];
		for (int i = 0; i < this.size; i++) {
			offsets[i + 1] = offsets[i] + binDict.decodeToBinary(i).length();
		}
		data = HostMemoryBuffer.allocate(offsets[this.size]);
		for (int i = 0; i < this.size; i++) {
			byte[] ba = binDict.decodeToBinary(i).getBytes();
			data.setBytes(offsets[i], ba, 0, ba.length);
		}
	}

	public HostMemoryBuffer getData() {
		return data;
	}

	public int[] getOffsets() {
		return offsets;
	}

	@Override
	public int getMaxId() {
		return this.size;
	}

	@Override
	public void close() {
		if (data != null) {
			data.close();
		}
	}

	private final int size;
	private final int[] offsets;
	private final HostMemoryBuffer data;

}
