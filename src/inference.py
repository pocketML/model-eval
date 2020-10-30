from time import sleep
import subprocess
import tracemalloc
import platform
import asyncio

class InferenceTask:
    TASK_SYSCALL = 1
    TASK_PROCESS = 2

    def __init__(self, task, task_type):
        self.task_type = task_type
        self.task = task

    def is_done(self):
        if self.task_type == InferenceTask.TASK_SYSCALL:
            # Check if exit code of process has been set.
            return self.task.poll() is not None
        elif self.task_type == InferenceTask.TASK_PROCESS:
            # Check if asyncio task is done.
            return self.task.exitcode is not None

def start_trace():
    # Task is run using python asyncio (in the current process) so we can use 'tracemalloc'
    tracemalloc.start()

def get_memory_snapshot(pid=None):
    if pid is not None:
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
    else:
        snapshot = tracemalloc.take_snapshot()
        statistics = snapshot.statistics("filename")
        dont_include = {"tracemalloc.py", "inference.py"}
        total_allocations = 0
        for stats in statistics:
            file_name = stats.traceback[0].filename.replace("\\", "/").split("/")[-1]
            if file_name not in dont_include:
                total_allocations += stats.size
        return total_allocations

async def monitor_inference(inference_task):
    """
    This method is called after a call to a model which will run an inference task.
    This method then records how much space the model takes up while running.
    """
    max_trace_count = 100

    print("Monitoring memory usage during inference...")

    memory_footprints = []
    pid = inference_task.task.pid

    while not inference_task.is_done():
        if (footprint := get_memory_snapshot(pid)) is not None:
            memory_footprints.append(footprint)
            if len(memory_footprints) > max_trace_count:
                memory_footprints = [max(memory_footprints)]
        await asyncio.sleep(0.5)

    print("Prediction/inference completed.")
    return max(memory_footprints)
