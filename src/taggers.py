from abc import ABC, abstractmethod
from glob import glob
import data_archives

class Tagger(ABC):
    def __init__(self, lang, treebank):
        self.epoch = 0
        self.lang = lang
        self.treebank = treebank

    def read_stdout(self, process_handler):
        if process_handler.poll() is not None:
            return None
        try:
            data = process_handler.stdout.readline()
        except KeyboardInterrupt:
            return None
        text = data.decode("utf-8")
        return text

    def inference_complete(self, process_handler):
        return process_handler.poll() is not None

    @abstractmethod
    def get_pred_acc(self):
        pass

    @abstractmethod
    def model_base_path(self):
        pass

    @abstractmethod
    def model_path(self):
        pass

    @abstractmethod
    def predict_path(self):
        pass

class SVMT(Tagger):
    ACC_STR = "TEST ACCURACY:"

    def __init__(self, lang, treebank):
        super().__init__(lang, treebank)
        cfg_path = "models/svmtool/bin/config.svmt"
        with open(cfg_path, "r", encoding="utf-8", newline=None) as fp:
            lines = fp.readlines()
        train_set = data_archives.get_dataset_path(lang, treebank, "train")
        test_set = data_archives.get_dataset_path(lang, treebank, "test")
        dev_set = data_archives.get_dataset_path(lang, treebank, "dev")
        lines[3] = f"TRAINSET = {train_set}\n"
        lines[5] = f"VALSET = {dev_set}\n"
        lines[7] = f"TESTSET = {test_set}\n"
        lines[11] = f"NAME = {self.model_base_path()}/pocketML\n"
        with open(cfg_path, "w", encoding="utf-8", newline="\n") as fp:
            for line in lines:
                fp.write(line)

    async def on_epoch_complete(self, process_handler):
        while (text := self.read_stdout(process_handler)) is not None:
            if (index := text.find(self.ACC_STR)) != -1:
                acc_str = text[index + len(self.ACC_STR) + 1:]
                if (pct_index := acc_str.find("%")) != -1:
                    self.epoch += 1
                    yield float(acc_str[:pct_index])

    def get_pred_acc(self):
        correct = 0
        total = 0
        with open(self.predict_path(), "r", encoding="utf-8") as fp:
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

    def model_base_path(self):
        return f"models/svmtool/pocketML/{self.lang}_{self.treebank}"

    def model_path(self):
        files = glob(f"{self.model_base_path()}/pocketML.FLD.*")
        if len(files) == 0:
            return self.model_base_path()
        split_files = [x.split(".") for x in files]
        split_files = [int(x[x.index("FLD") + 1]) for x in split_files]
        split_files.sort()
        return f"{self.model_base_path()}/pocketML.FLD.{split_files[-1]}"

    def predict_path(self):
        return f"{self.model_base_path()}/preds.out"

class BILSTM(Tagger):
    ACC_STR = "dev accuracy:"
    MODEL_BASE_PATH = "models/bilstm-aux/pocketML"

    async def on_epoch_complete(self, process_handler):
        while (text := self.read_stdout(process_handler)) is not None:
            if (index := text.find(self.ACC_STR)) != -1:
                acc_str = text[index + len(self.ACC_STR) + 1:]
                self.epoch += 1
                yield float(acc_str) * 100

    def get_pred_acc(self):
        correct = 0
        total = 0
        with open(self.predict_path(), "r", encoding="utf-8") as fp:
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

    def model_base_path(self):
        return f"models/bilstm-aux/pocketML/{self.lang}_{self.treebank}"

    def model_path(self):
        return f"{self.model_base_path()}/pocketML"

    def predict_path(self):
        return f"{self.model_base_path()}/preds.out.task0"

class POSADV(Tagger):
    ACC_STR = "dev accuracy:"

    async def on_epoch_complete(self, process_handler):
        while (text := self.read_stdout(process_handler)) is not None:
            print(text)
            if (index := text.find(self.ACC_STR)) != -1:
                acc_str = text[index + len(self.ACC_STR) + 1:]
                self.epoch += 1
                yield float(acc_str) * 100

    def get_pred_acc(self):
        correct = 0
        total = 0
        with open(self.predict_path(), "r", encoding="utf-8") as fp:
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

    def model_base_path(self):
        return f"models/pos_adv/pocketML/pos_{self.lang}_{self.treebank}"

    def model_path(self):
        return f"{self.model_base_path()}/pocketML"

    def predict_path(self):
        return f"{self.model_base_path()}/preds.out.task0"
