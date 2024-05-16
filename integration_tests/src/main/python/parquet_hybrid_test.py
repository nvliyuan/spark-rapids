# Copyright (c) 2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import pytest

from asserts import *
from data_gen import *
from marks import *
from pyspark.sql.types import *
from pyspark.sql.functions import *
from spark_session import *


def read_parquet_df(data_path):
    return lambda spark : spark.read.parquet(data_path)

def read_parquet_sql(data_path):
    return lambda spark : spark.sql('select * from parquet.`{}`'.format(data_path))


rebase_write_corrected_conf = {
    'spark.sql.legacy.parquet.datetimeRebaseModeInWrite': 'CORRECTED',
    'spark.sql.legacy.parquet.int96RebaseModeInWrite': 'CORRECTED'
}

rebase_write_legacy_conf = {
    'spark.sql.legacy.parquet.datetimeRebaseModeInWrite': 'LEGACY',
    'spark.sql.legacy.parquet.int96RebaseModeInWrite': 'LEGACY'
}

# Like the standard map_gens_sample but with timestamps limited
parquet_map_gens = [MapGen(f(nullable=False), f()) for f in [
    BooleanGen, ByteGen, ShortGen, IntegerGen, LongGen, FloatGen, DoubleGen, DateGen,
    lambda nullable=True: TimestampGen(start=datetime(1900, 1, 1, tzinfo=timezone.utc), nullable=nullable)]] + \
                   [simple_string_to_string_map_gen,
                    MapGen(StringGen(pattern='key_[0-9]', nullable=False), ArrayGen(string_gen), max_length=10),
                    MapGen(RepeatSeqGen(IntegerGen(nullable=False), 10), long_gen, max_length=10),
                    MapGen(StringGen(pattern='key_[0-9]', nullable=False), simple_string_to_string_map_gen)]

parquet_gens_list = [
    [
        byte_gen, short_gen, int_gen, long_gen, float_gen, double_gen,
        string_gen, boolean_gen, date_gen,
        TimestampGen(start=datetime(1900, 1, 1, tzinfo=timezone.utc)), ArrayGen(byte_gen),
        ArrayGen(long_gen), ArrayGen(string_gen), ArrayGen(date_gen),
        ArrayGen(TimestampGen(start=datetime(1900, 1, 1, tzinfo=timezone.utc))),
        ArrayGen(ArrayGen(byte_gen)),
        StructGen([['child0', ArrayGen(byte_gen)], ['child1', byte_gen], ['child2', float_gen]]),
        ArrayGen(StructGen([['child0', string_gen], ['child1', double_gen], ['child2', int_gen]]))
    ] + parquet_map_gens
]

pattern = "[a-z0-9A-Z]{1,10}"
array_string_debug = [
    [ArrayGen(StringGen(pattern=pattern, nullable=False), nullable=False, max_length=3)],
    [ArrayGen(StringGen(pattern=pattern, nullable=(True, 50.0)), nullable=False, max_length=3)],
    [ArrayGen(StringGen(pattern=pattern, nullable=False), nullable=(True, 50.0), max_length=3)],
    [ArrayGen(StringGen(pattern=pattern, nullable=(True, 50.0)), nullable=(True, 50.0), max_length=3)],
    [ArrayGen(StringGen(pattern=pattern, nullable=False), nullable=False)],
    [ArrayGen(StringGen(pattern=pattern, nullable=(True, 50.0)), nullable=False)],
    [ArrayGen(StringGen(pattern=pattern, nullable=False), nullable=(True, 50.0))],
    [ArrayGen(StringGen(pattern=pattern, nullable=(True, 50.0)), nullable=(True, 50.0))],
    [StringGen(pattern=pattern, nullable=False)],
    [StringGen(pattern=pattern, nullable=(True, 50.0))],
]

map_simple_debug = [[g] for g in all_basic_map_gens] + [
    [MapGen(StringGen(pattern='key_[0-9]', nullable=False),
            StringGen(nullable=(True, 50.0)),
            max_length=3,
            nullable=(True, 50.0))
     ]
]

map_nested_debug = [
    [ArrayGen(ArrayGen(LongGen(), max_length=5), max_length=5)],
    [ArrayGen(StringGen(pattern=pattern), max_length=5)],
    [ArrayGen(ArrayGen(StringGen(pattern=pattern), max_length=3, nullable=False), max_length=3, nullable=False)],
    [ArrayGen(ArrayGen(StringGen(pattern=pattern), max_length=3), max_length=3)],
    [MapGen(StringGen(pattern='key_[0-9]', nullable=False), ArrayGen(StringGen(pattern=pattern)), max_length=10)],
    [MapGen(RepeatSeqGen(IntegerGen(nullable=False), 10), long_gen, max_length=10)],
    [MapGen(StringGen(pattern='key_[0-9]', nullable=False), simple_string_to_string_map_gen)],
    [MapGen(IntegerGen(False), ArrayGen(int_gen, max_length=3), max_length=3)],
    [MapGen(ByteGen(False), MapGen(FloatGen(False), date_gen, max_length=3), max_length=3)],
]

struct_debug = [
    [all_basic_struct_gen],
]



# test with original parquet file reader, the multi-file parallel reader for cloud, and coalesce file reader for
# non-cloud
original_parquet_file_reader_conf = {'spark.rapids.sql.format.parquet.reader.type': 'PERFILE'}
multithreaded_parquet_file_reader_conf = {'spark.rapids.sql.format.parquet.reader.type': 'MULTITHREADED',
                                          'spark.rapids.sql.reader.multithreaded.combine.sizeBytes': '0',
                                          'spark.rapids.sql.reader.multithreaded.read.keepOrder': True}
coalesce_parquet_file_reader_conf = {'spark.rapids.sql.format.parquet.reader.type': 'COALESCING'}
coalesce_parquet_file_reader_multithread_filter_chunked_conf = {'spark.rapids.sql.format.parquet.reader.type': 'COALESCING',
                                                                'spark.rapids.sql.coalescing.reader.numFilterParallel': '2',
                                                                'spark.rapids.sql.reader.chunked': True}
coalesce_parquet_file_reader_multithread_filter_conf = {'spark.rapids.sql.format.parquet.reader.type': 'COALESCING',
                                                        'spark.rapids.sql.coalescing.reader.numFilterParallel': '2',
                                                        'spark.rapids.sql.reader.chunked': False}
native_parquet_file_reader_conf = {'spark.rapids.sql.format.parquet.reader.type': 'PERFILE',
                                   'spark.rapids.sql.format.parquet.reader.footer.type': 'NATIVE'}
native_multithreaded_parquet_file_reader_conf = {'spark.rapids.sql.format.parquet.reader.type': 'MULTITHREADED',
                                                 'spark.rapids.sql.format.parquet.reader.footer.type': 'NATIVE',
                                                 'spark.rapids.sql.reader.multithreaded.combine.sizeBytes': '0',
                                                 'spark.rapids.sql.reader.multithreaded.read.keepOrder': True}
native_coalesce_parquet_file_reader_conf = {'spark.rapids.sql.format.parquet.reader.type': 'COALESCING',
                                            'spark.rapids.sql.format.parquet.reader.footer.type': 'NATIVE'}
native_coalesce_parquet_file_reader_chunked_conf = {'spark.rapids.sql.format.parquet.reader.type': 'COALESCING',
                                                    'spark.rapids.sql.format.parquet.reader.footer.type': 'NATIVE',
                                                    'spark.rapids.sql.reader.chunked': True}
combining_multithreaded_parquet_file_reader_conf_ordered = {'spark.rapids.sql.format.parquet.reader.type': 'MULTITHREADED',
                                                            'spark.rapids.sql.reader.multithreaded.combine.sizeBytes': '64m',
                                                            'spark.rapids.sql.reader.multithreaded.read.keepOrder': True}
combining_multithreaded_parquet_file_reader_conf_unordered = pytest.param({'spark.rapids.sql.format.parquet.reader.type': 'MULTITHREADED',
                                                                           'spark.rapids.sql.reader.multithreaded.combine.sizeBytes': '64m',
                                                                           'spark.rapids.sql.reader.multithreaded.read.keepOrder': False}, marks=pytest.mark.ignore_order(local=True))
combining_multithreaded_parquet_file_reader_deprecated_conf_ordered = {
    'spark.rapids.sql.format.parquet.reader.type': 'MULTITHREADED',
    'spark.rapids.sql.format.parquet.multithreaded.combine.sizeBytes': '64m',
    'spark.rapids.sql.format.parquet.multithreaded.read.keepOrder': True}


# For now the native configs are not compatible with spark.sql.parquet.writeLegacyFormat written files
# for nested types
reader_opt_confs_native = [native_multithreaded_parquet_file_reader_conf]

reader_opt_confs_no_native = [multithreaded_parquet_file_reader_conf,
                              combining_multithreaded_parquet_file_reader_conf_ordered,
                              combining_multithreaded_parquet_file_reader_deprecated_conf_ordered]

reader_opt_confs = reader_opt_confs_native + reader_opt_confs_no_native

hybrid_opt_confs = [
    {
        'spark.rapids.sql.format.parquet.reader.type': 'MULTITHREADED',
        'spark.rapids.sql.format.parquet.reader.footer.type': 'NATIVE',
        'spark.rapids.sql.parquet.scan.hybridMode': 'CPU_ONLY',
        'spark.rapids.sql.parquet.scan.hostParallelism': 2,
        'spark.rapids.sql.parquet.scan.hostBatchSizeBytes': bs,
        'spark.rapids.sql.parquet.scan.async': "true",
        'spark.rapids.sql.parquet.scan.enableDictLateMat': dict_lat_mat,
        "spark.rapids.sql.test.injectRetryOOM": "true",
        "spark.rapids.sql.parquet.scan.unsafeDecompress": unsafe_decompress
    }
    for unsafe_decompress in ["true", "false"]
    for dict_lat_mat in ["false", "true"]
    for bs in [256, 256 << 3, 256 << 6, 256 << 10, 256 << 12, 256 << 14]
]


@pytest.mark.parametrize('parquet_gens', parquet_gens_list, ids=idfn)
@pytest.mark.parametrize('read_func', [read_parquet_df])
@pytest.mark.parametrize('reader_confs', hybrid_opt_confs)
@pytest.mark.parametrize('v1_enabled_list', ["parquet", ""])
@pytest.mark.parametrize('length', [10, 100, 500, 1000, 2000])
@pytest.mark.parametrize('parquet_codec', ["zstd", "snappy", "uncompressed"])
@pytest.mark.parametrize('parquet_block_size', [1048576 * 8, 1048576 * 32, 1048576 * 128])
@tz_sensitive_test
@allow_non_gpu(*non_utc_allow)
def test_parquet_read_round_trip(spark_tmp_path,
                                 parquet_gens,
                                 read_func,
                                 reader_confs,
                                 v1_enabled_list,
                                 length,
                                 parquet_codec,
                                 parquet_block_size):
    gen_list = [('_c' + str(i), gen) for i, gen in enumerate(parquet_gens)]
    data_path = spark_tmp_path + '/PARQUET_DATA'
    with_cpu_session(
        lambda spark: gen_df(spark, gen_list, length=length, seed=int(time.time() * 1000))
        .write
        .option("parquet.block.size", parquet_block_size)
        .option("parquet.page.size", 1048576)
        .option("parquet.enable.dictionary", True)
        .option("parquet.dictionary.page.size", 1048576 * 128)
        .option("parquet.compression", parquet_codec)
        .parquet(data_path),
        conf=rebase_write_corrected_conf)
    all_confs = copy_and_update(reader_confs, {
        'spark.sql.sources.useV1SourceList': v1_enabled_list,
        # set the int96 rebase mode values because its LEGACY in databricks which will preclude this op from running on GPU
        'spark.sql.legacy.parquet.int96RebaseModeInRead' : 'CORRECTED',
        'spark.sql.legacy.parquet.datetimeRebaseModeInRead': 'CORRECTED'})
    # once https://github.com/NVIDIA/spark-rapids/issues/1126 is in we can remove spark.sql.legacy.parquet.datetimeRebaseModeInRead config which is a workaround
    # for nested timestamp/date support
    assert_gpu_and_cpu_are_equal_collect(read_func(data_path), conf=all_confs)
