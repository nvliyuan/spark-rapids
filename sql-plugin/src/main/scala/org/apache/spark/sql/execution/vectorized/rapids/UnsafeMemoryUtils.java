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

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;

public class UnsafeMemoryUtils {

	static Method copyMemoryDirectly;

	static {
		try {
			Class<?> clz = Class.forName("ai.rapids.cudf.UnsafeMemoryAccessor");
			copyMemoryDirectly = clz.getMethod("copyMemory",
					Object.class, long.class, Object.class, long.class, long.class);
			copyMemoryDirectly.setAccessible(true);
		} catch (ClassNotFoundException | NoSuchMethodException e) {
			throw new RuntimeException(e);
		}
	}

	/**
	 * The reflection of `ai.rapids.cudf.UnsafeMemoryAccessor.copyMemory`
	 * Copy memory from one address to the other.
	 */
	public static void copyMemory(Object src, long srcOffset, Object dst, long dstOffset,
																long length) {
		try {
			copyMemoryDirectly.invoke(null,
					src,
					srcOffset,
					dst,
					dstOffset,
					length);
		} catch (IllegalAccessException | InvocationTargetException e) {
			throw new RuntimeException(e);
		}
	}

}
