This doc describes how to dump an Exec runtime meta and column batch data to a directory,
and replay the Exec with the dumped meta and data. When encountering a perf issue using this tool
can dump the runtime meta/data and then replay. It will spend less time when collecting NSYS and 
NCU information when replaying the dumped runtime meta/data.

# Dump a Exec meta and a column batch data

## compile Spark-Rapids jar
e.g.: compile for Spark330
```
mvn clean install -DskipTests -pl dist -am -DallowConventionalDistJar=true -Dbuildver=330 
```

## enable dump
This assumes that the RAPIDs Accelerator has already been enabled.   
This feature is disabled by default.   
Set the following configurations to enable this feature:   

``` 
spark.conf.set("spark.rapids.sql.test.replay.exec.type", "project")
```
Default `type` value is empty which means do not dump.   
Set this `type` to `project` if you want to dump Project Exec runtime data. Currently only support
`project` and empty.

```
spark.conf.set("spark.rapids.sql.test.replay.exec.dumpDir", "file:/tmp")
```
Default value is `file:/tmp`.  
specify the dump directory, e.g.: `file:/tmp/my-debug-path`.  
This path should be a directory or someplace that we can create a directory to
store files in. Remote path is supported e.g.: `hdfs://url:9000/path/to/save`

```
spark.conf.set("spark.rapids.sql.test.replay.exec.threshold.timeMS", 1000)
```
Default value is 1000MS.   
Only dump the column batches when it's executing time exceeds threshold time. 

```  
spark.conf.set("spark.rapids.sql.test.replay.exec.batch.limit", 1)
```
This config defines the max dumping number of column batches.   
Default value is 1.

## run a Spark job 
This dumping only happens when GPU Exec is running, so should enable Spark-Rapids.
After the job is done, check the dump path will have files like:
```
/tmp/replay-exec:
  - xxx_GpuTieredProject.meta      // this is serialized GPU Tiered Project case class  
  - xxx_cb_types.meta              // this is types for column batch
  - xxx_cb_data_0_101656570.parquet  // this is data for column batch
```
The prefix `xxx` is the hash code for a specific Project Exec, this is used to distinguish multiple
Project Execs.

# Replay saved Exec runtime meta and data

## Set environment variable
```
export PLUGIN_JAR=path_to_spark_rapids_jar
export SPARK_HOME=path_to_spark_home_330
```

## replay command

### Collect NSYS with replaying
```
nsys profile $SPARK_HOME/bin/spark-submit \
  --class com.nvidia.spark.rapids.test.ProjectExecReplayer \
  --conf spark.rapids.sql.explain=ALL \
  --master local[*] \
  --jars ${PLUGIN_JAR} \
  ${PLUGIN_JAR} <path_to_saved_replay_dir> <hash_code_of_project_exec>
```

<path_to_saved_replay_dir> is the replay directory   
<hash_code_of_project_exec> is the hash code for a specific Project Exec. Dumping may generate
multiple data for each project, here should specify replay which project.   
The replaying only replay one column batch, if there are multiple column batches for a project, it
will only replay the first column batch.
