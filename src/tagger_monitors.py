import asyncio

class Parser:
    def __init__(self, pipe):
        self.pipe = pipe

    async def on_epoch_complete(self):
        data = self.pipe.read1()
        text = data.decode("utf-8")
        return text

    async def on_evaluation_complete(self):
        pass

class SVMTParser(Parser):
    ACC_STR = "TEST ACCURACY:"

    async def on_epoch_complete(self):
        while (text := await super().on_epoch_complete()) != "":
            if (index := text.find(self.ACC_STR)) != -1:
                acc_str = text[index + len(self.ACC_STR) + 1:]
                if (pct_index := acc_str.find("%")) != -1:
                    yield float(acc_str[:pct_index])

class BILSTMParser(Parser):
    ACC_STR = "TEST ACCURACY:"

    async def on_epoch_complete(self):
        while (text := await super().on_epoch_complete()) != "":
            yield text
            # if (index := text.find(self.ACC_STR)) != -1:
            #     acc_str = text[index + len(self.ACC_STR) + 1:]
            #     if (pct_index := acc_str.find("%")) != -1:
            #         yield float(acc_str[:pct_index])
