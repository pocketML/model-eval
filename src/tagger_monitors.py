import asyncio
from subprocess import TimeoutExpired

class Parser:
    def __init__(self, process_handler):
        self.epoch = 0
        self.process_handler = process_handler

    async def on_epoch_complete(self):
        if self.process_handler.poll() is not None:
            return None
        try:
            data = self.process_handler.stdout.readline()
        except KeyboardInterrupt:
            return None
        text = data.decode("utf-8")
        return text

    async def on_evaluation_complete(self):
        pass

class SVMTParser(Parser):
    ACC_STR = "TEST ACCURACY:"

    async def on_epoch_complete(self):
        while (text := await super().on_epoch_complete()) is not None:
            if (index := text.find(self.ACC_STR)) != -1:
                acc_str = text[index + len(self.ACC_STR) + 1:]
                if (pct_index := acc_str.find("%")) != -1:
                    self.epoch += 1
                    yield float(acc_str[:pct_index])

class BILSTMParser(Parser):
    ACC_STR = "dev accuracy:"

    async def on_epoch_complete(self):
        while (text := await super().on_epoch_complete()) is not None:
            if (index := text.find(self.ACC_STR)) != -1:
                acc_str = text[index + len(self.ACC_STR) + 1:]
                self.epoch += 1
                yield float(acc_str) * 100
