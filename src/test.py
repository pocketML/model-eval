import threading
import io
import sys
import time

def monitor(out_thing, old_stdout):
    while True:
        text = out_thing.getvalue()
        if text != "":
            sys.stdout = old_stdout
            print(text)
            sys.stdout = out_thing

out_thing = io.StringIO()
threading.Thread(target=monitor, args=(out_thing, sys.stdout)).start()
print("HEY!")

time.sleep(0.2)
print("MORE!!!!")
