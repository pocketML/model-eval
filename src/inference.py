import tracemalloc

async def monitor_inference(monitor, model_name, use_loadbar=True):
    tracemalloc.start()
    print("Current: %d, Peak %d" % tracemalloc.get_traced_memory())
