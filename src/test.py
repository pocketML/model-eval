from time import sleep
import tracemalloc

tracemalloc.start()

stuff = [x for x in range(int(10e05))]

print(len(stuff))

sleep(0.5)

snapshot = tracemalloc.take_snapshot()
statistics = snapshot.statistics("filename")

print(__file__)

for allocs_by_file in statistics:
    print(allocs_by_file.traceback[0].filename)

print(statistics.size / 1000)
