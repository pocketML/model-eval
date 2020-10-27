class Tagger:
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

    def inference_complete(self):
        return self.process_handler.poll() is not None

    def get_pred_acc(self):
        pass

class SVMT(Tagger):
    ACC_STR = "TEST ACCURACY:"

    async def on_epoch_complete(self):
        while (text := await super().on_epoch_complete()) is not None:
            if (index := text.find(self.ACC_STR)) != -1:
                acc_str = text[index + len(self.ACC_STR) + 1:]
                if (pct_index := acc_str.find("%")) != -1:
                    self.epoch += 1
                    yield float(acc_str[:pct_index])

    def get_pred_acc(self):
        pass

class BILSTM(Tagger):
    ACC_STR = "dev accuracy:"

    async def on_epoch_complete(self):
        while (text := await super().on_epoch_complete()) is not None:
            if (index := text.find(self.ACC_STR)) != -1:
                acc_str = text[index + len(self.ACC_STR) + 1:]
                self.epoch += 1
                yield float(acc_str) * 100

    def get_pred_acc(self):
        pass
