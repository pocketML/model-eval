import subprocess
import platform
import asyncio

def get_memory_snapshot(pid):
    """
    Grab a snapshot of how much memory the inference process is currently using.
    """
    if platform.system() == "Windows":
        cmd = f"tasklist /fi \"pid eq {pid}\" | findstr \" K$"
    else:
        cmd = f"pmap {pid} " + "| tail -n 1 | awk '/[0-9]K/{print $2}'"

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
    data = process.stdout.read()
    text = ""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError: # Apparently this happens sometimes.
        print("Unicode decode error during memory snapshot.")
        return None
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
        return int(text.strip()[:-1]) # Usage in kilobytes.

def process_is_alive(process):
    if isinstance(process, subprocess.Popen):
        return process.poll() is None # Handle subprocess 'Popen' class.
    return process.exitcode is None # Handle multiprocess 'Process' class.

async def monitor_inference(model, process):
    """
    This method is called after a call to a model which will run an inference task.
    This method then records how much space the model takes up while running.
    """
    max_trace_count = 100

    print("Monitoring memory usage during inference...")

    memory_footprints = []
    pid = process.pid
    while process_is_alive(process):
        if (footprint := get_memory_snapshot(pid)) is not None:
            memory_footprints.append(footprint)
            if len(memory_footprints) > max_trace_count:
                memory_footprints = [max(memory_footprints)]
        await asyncio.sleep(0.5)

    code_size = model.code_size() // 1000
    model_size = model.model_size() // 1000
    if not model.compressed_model_exists():
        print("Compressing trained model...")
        model.compress_model()

    compressed_size = model.compressed_model_size() // 1000

    print("Prediction/inference completed.")
    # Return all our different size measurements.
    return max(memory_footprints), code_size, model_size, compressed_size
