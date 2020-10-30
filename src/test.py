import subprocess
import random
import threading
import platform
import tracemalloc
from os import getpid
from time import sleep
from math import log

def bytes_needed(n):
    if n == 0:
        return 1
    return int(log(n, 256)) + 1

def do_work():
    mean_items = 1000000
    variance = int(mean_items * 1.5)
    max_items = mean_items + variance
    min_items = mean_items - variance

    stored_stuff = [
        x for x in range(mean_items)
    ]

    while True:
        cohicei = random.random() > 0.5
        if cohicei and len(stored_stuff) < max_items:
            stored_stuff.append(random.randint(0, 100000))
        elif not cohicei and len(stored_stuff) > min_items:
            stored_stuff.pop()
        sleep(0.01)

def main():
    tracemalloc.start(25)
    threading.Thread(target=do_work).start()

    pid = getpid()
    if platform.system() == "Windows":
        cmd = f"tasklist /fi \"pid eq {pid}\" | findstr \" K$"
    else:
        cmd = f"pmap {pid} " + "| tail -n 1 | awk '/[0-9]K/{print $2}'"
    print(f"Process id: {pid}", flush=True)
    print(f"Command: {cmd}", flush=True)

    while True:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
        data = process.stdout.read()
        print(data)
        text = data.decode("utf-8")
        split = data.decode("utf-8").split(" ")
        text = split[-2].replace(".", "")
        working_size = text
        print(working_size + "KB", flush=True)
        print(tracemalloc.get_traced_memory()[0])
        sleep(1)

try:
    main()
except KeyboardInterrupt:
    exit(0)
