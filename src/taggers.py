from glob import glob

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
    SAVED_MODEL = "models/svmtool/pocketML"
    PREDICTIONS = "models/svmtool/pocketML/preds.out"

    async def on_epoch_complete(self):
        while (text := await super().on_epoch_complete()) is not None:
            if (index := text.find(self.ACC_STR)) != -1:
                acc_str = text[index + len(self.ACC_STR) + 1:]
                if (pct_index := acc_str.find("%")) != -1:
                    self.epoch += 1
                    yield float(acc_str[:pct_index])

    def get_pred_acc(self):
        correct = 0
        total = 0
        with open(SVMT.PREDICTIONS, "r", encoding="utf-8") as fp:
            lines = fp.readlines()
            for line in lines:
                line = line.strip()
                if line == "":
                    continue
                total += 1
                split = line.split(None)
                predicted = split[1]
                actual = split[2]
                if predicted == actual:
                    correct += 1
        return correct / total

    @staticmethod
    def latest_model():
        files = glob(f"{SVMT.SAVED_MODEL}/pocketML.FLD.*")
        split_files = [x.split(".") for x in files]
        split_files = [int(x[x.index("FLD") + 1]) for x in split_files]
        split_files.sort()
        return f"pocketML.FLD.{split_files[-1]}"

class BILSTM(Tagger):
    ACC_STR = "dev accuracy:"
    SAVED_MODEL = "models/bilstm-aux/en"
    PREDICTIONS = "models/bilstm-aux/preds.out"

    async def on_epoch_complete(self):
        while (text := await super().on_epoch_complete()) is not None:
            if (index := text.find(self.ACC_STR)) != -1:
                acc_str = text[index + len(self.ACC_STR) + 1:]
                self.epoch += 1
                yield float(acc_str) * 100

    def get_pred_acc(self):
        correct = 0
        total = 0
        with open(BILSTM.PREDICTIONS + ".task0", "r", encoding="utf-8") as fp:
            lines = fp.readlines()
            index = 0
            while index < len(lines):
                line = lines[index].strip()
                index += 1
                if line == "":
                    continue
                actual = lines[index].strip()
                index += 1
                total += 1
                split = line.split(None)
                predicted = split[1]
                if predicted == actual:
                    correct += 1
        return correct / total

class POSADV(Tagger):
    ACC_STR = "dev accuracy:"

    async def on_epoch_complete(self):
        while (text := await super().on_epoch_complete()) is not None:
            print(text)
            if (index := text.find(self.ACC_STR)) != -1:
                acc_str = text[index + len(self.ACC_STR) + 1:]
                self.epoch += 1
                yield float(acc_str) * 100
