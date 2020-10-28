import tracemalloc
import asyncio

class InferenceTask:
    TASK_SYSCALL = 1
    TASK_ASYNCIO = 2

    def __init__(self, task, task_type):
        self.task_type = task_type
        self.task = task

    def is_done(self):
        if self.task_type == InferenceTask.TASK_SYSCALL:
            # Check if exit code of process has been set.
            return self.task.poll() is not None
        elif self.task_type == InferenceTask.TASK_ASYNCIO:
            # Check if asyncio task is done.
            return self.task.done()

async def monitor_inference(inference_task):
    """
    This method is called after a call to a model which will run an inference task.
    This method then records how much space the model takes up while running.
    """
    print("Monitoring memory usage during inference...")
    tracemalloc.start()
    while not inference_task.is_done():
        # Record disk/memory usage of the thing...
        await asyncio.sleep(0.1)
    print("Prediction/inference completed.")
    return tracemalloc.get_traced_memory()
