import tracemalloc
import asyncio

async def monitor_inference(monitor):
    """
    This method is called after a call to a model which will run an inference task.
    This method then records how much space the model takes up while running.
    """
    tracemalloc.start()
    while not monitor.inference_complete():
        # Record disk/memory usage of the thing...
        await asyncio.sleep(0.1)
    return tracemalloc.get_traced_memory()
