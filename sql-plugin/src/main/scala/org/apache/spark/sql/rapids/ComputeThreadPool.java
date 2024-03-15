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

package org.apache.spark.sql.rapids;

import java.util.Comparator;
import java.util.concurrent.*;

public class ComputeThreadPool {

	public enum TaskStatus {
		PENDING,
		RUNNING,
		SUCCESSFUL,
		FAILED
	}

	public interface FailedCallback {
		void callback(Throwable ex);
	}

	public interface SuccessfulCallback<Result> {
		void callback(Result result);
	}


	public static class TaskWithPriority<Result> {

		public TaskWithPriority(Callable<Result> task, int priority, boolean semaphoreAcquired) {
			this(task, priority, semaphoreAcquired, null, null);
		}

		public TaskWithPriority(Callable<Result> task,
														int priority,
														boolean semaphoreAcquired,
														SuccessfulCallback<Result> succeedCb,
														FailedCallback failedCb) {
			this.task = task;
			this.priority = priority;
			this.promise = new CompletableFuture<>();
			this.status = TaskStatus.PENDING;
			this.semaphoreAcquired = semaphoreAcquired;
			this.succeedCb = succeedCb;
			this.failedCb = failedCb;
		}

		private void run() throws Exception {
			try {
				Result ret = task.call();
				promise.complete(ret);
				if (succeedCb != null) {
					succeedCb.callback(ret);
				}
			} catch (Throwable e) {
				promise.completeExceptionally(e);
				if (failedCb != null) {
					failedCb.callback(e);
				}
				throw e;
			}
		}

		public TaskStatus status() {
			return this.status;
		}

		public Future<Result> future() {
			return promise;
		}

		public Result getResult() throws ExecutionException, InterruptedException {
			return promise.get();
		}

		public void cancel() {
			if (status == TaskStatus.PENDING) {
				if (!ComputeThreadPool.removeTask(this)) {
					promise.cancel(true);
				}
			} else if (status == TaskStatus.RUNNING) {
				promise.cancel(true);
			}
		}

		private final CompletableFuture<Result> promise;
		private final Callable<Result> task;
		private final int priority;
		private volatile TaskStatus status;
		private final boolean semaphoreAcquired;

		private final SuccessfulCallback<Result> succeedCb;
		private final FailedCallback failedCb;
	}

	private ComputeThreadPool(int threadNum, int taskQueueCapacity) {
		this.taskQueue = new PriorityBlockingQueue<>(
				taskQueueCapacity, Comparator.comparingInt(o -> o.priority));
		ThreadGroup threadGroup = new ThreadGroup("CPU_INTENSIVE_THREADS");
		this.workerSemaphore = new Semaphore(threadNum);

		workers = new Worker[threadNum];
		for (int i = 0; i < threadNum; ++i) {
			workers[i] = new Worker(threadGroup, workerSemaphore, taskQueue);
			workers[i].start();
		}
	}

	private static class Worker extends Thread {

		Worker(ThreadGroup threadGroup, Semaphore semaphore, PriorityBlockingQueue<TaskWithPriority<?>> taskQueue) {
			super(threadGroup, Worker.getRunner(semaphore, taskQueue));
		}

		private static Runnable getRunner(Semaphore semaphore, PriorityBlockingQueue<TaskWithPriority<?>> taskQueue) {
			return new Runnable() {
				private final Semaphore sph = semaphore;
				private final PriorityBlockingQueue<TaskWithPriority<?>> queue = taskQueue;
				@Override
				public void run() {
					System.err.println("background thread started...");
					while (true) {
						try {
							TaskWithPriority<?> task = queue.take();
							task.status = TaskStatus.RUNNING;
							if (!task.semaphoreAcquired) {
								sph.acquire();
							}
							try {
								task.run();
								task.status = TaskStatus.SUCCESSFUL;
							} catch (Exception ex) {
								task.status = TaskStatus.FAILED;
								StringBuilder stackTrace = new StringBuilder();
								stackTrace.append(ex).append('\n');
								for (StackTraceElement elem : ex.getStackTrace()) {
									stackTrace.append(elem.toString()).append('\n');
								}
								System.err.println("ComputeThreadPool: background task failed: " + stackTrace);
							}
						} catch (InterruptedException e) {
							throw new RuntimeException(e);
						} finally {
							sph.release();
						}
					}
				}
			};
		}
	}

	public void close() {
		for (Worker worker : workers) {
			if (worker.isAlive()) {
				worker.interrupt();
			}
		}
	}

	public static void submitTask(TaskWithPriority<?> task) {
		INSTANCE.taskQueue.add(task);
	}

	public static synchronized void launch(int threadNum, int taskQueueCapacity) {
		if (INSTANCE == null) {
			INSTANCE = new ComputeThreadPool(threadNum, taskQueueCapacity);
		}
	}

	public static boolean bookIdleWorker(long timeout) {
		try {
			return INSTANCE.workerSemaphore.tryAcquire(timeout, TimeUnit.MILLISECONDS);
		} catch (InterruptedException e) {
			throw new RuntimeException(e);
		}
	}

	public static int releaseWorker() {
		INSTANCE.workerSemaphore.release();
		return INSTANCE.workerSemaphore.availablePermits();
	}

	public static ComputeThreadPool getInstance() {
		if (INSTANCE == null) {
			throw new RuntimeException("CpuIntensiveThreadPool is NOT initialized");
		}
		return INSTANCE;
	}

	public static boolean removeTask(TaskWithPriority<?> task) {
		if (INSTANCE == null) {
			throw new RuntimeException("CpuIntensiveThreadPool is NOT initialized");
		}
		return INSTANCE.taskQueue.remove(task);
	}

	private static ComputeThreadPool INSTANCE = null;

	private final Semaphore workerSemaphore;
	private final PriorityBlockingQueue<TaskWithPriority<?>> taskQueue;
	private final Worker[] workers;

}
