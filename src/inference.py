import subprocess
import tracemalloc
import platform
import asyncio

from nltk.util import Index

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

def shell_memory_snapshot(pid):
    if platform.system() == "Windows":
        cmd = f"tasklist /fi \"pid eq {pid}\" | findstr \" K$"
    else:
        cmd = f"pmap {pid} " + "| tail -n 1 | awk '/[0-9]K/{print $2}'"

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
    data = process.stdout.read()
    text = data.decode("utf-8")
    if text == "":
        return None

    if platform.system() == "Windows":
        split = text.split(" ")
        try:
            kb = split[-2].replace(".", "")
            return int(kb)
        except IndexError:
            print(f"Error: {text}")
            return None
    else:
        return int(text[:-1]) # Usage in kilobytes.

async def monitor_inference(inference_task):
    """
    This method is called after a call to a model which will run an inference task.
    This method then records how much space the model takes up while running.
    """
    print("Monitoring memory usage during inference...")
    if inference_task.task_type == InferenceTask.TASK_ASYNCIO:
        # Task is run using python asyncio (in the current process) so we can use 'tracemalloc'
        tracemalloc.start()

    shell_footprints = []
    while not inference_task.is_done():
        if inference_task.task_type == InferenceTask.TASK_SYSCALL:
            if (footprint := shell_memory_snapshot(inference_task.task.pid)) is not None:
                shell_footprints.append(footprint)
        await asyncio.sleep(0.1)

    print("Prediction/inference completed.")
    footprint = ((tracemalloc.get_traced_memory()[1] - tracemalloc.get_tracemalloc_memory()) // 1024
                 if inference_task.task_type == InferenceTask.TASK_ASYNCIO
                 else max(shell_footprints))
    return footprint
