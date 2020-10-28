from abc import ABC, abstractmethod
from glob import glob

from nltk import tree
import data_archives

class Tagger(ABC):
    def __init__(self, args):
        self.epoch = 0
        self.lang = args.lang
        self.treebank = args.treebank
        self.iters = args.iter

    def read_stdout(self, process_handler):
        if process_handler.poll() is not None:
            return None
        data = process_handler.stdout.readline()
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

    def __init__(self, args):
        super().__init__(args)
        cfg_path = "models/svmtool/bin/config.svmt"
        with open(cfg_path, "r", encoding="utf-8", newline=None) as fp:
            lines = fp.readlines()
        train_set = data_archives.get_dataset_path(self.lang, self.treebank, "train")
        test_set = data_archives.get_dataset_path(self.lang, self.treebank, "test")
        dev_set = data_archives.get_dataset_path(self.lang, self.treebank, "dev")
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

    async def on_epoch_complete(self, process_handler):
        while (text := self.read_stdout(process_handler)) is not None:
            if (index := text.find(self.ACC_STR)) != -1:
                acc_str = text[index + len(self.ACC_STR) + 1:]
                self.epoch += 1
                yield float(acc_str)

    def get_pred_acc(self):
        correct = 0
        total = 0
        with open(self.predict_path() + ".task0", "r", encoding="utf-8") as fp:
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
        return f"{self.model_base_path()}/preds.out"

class POSADV(Tagger):
    ACC_STR = "test loss:"

    async def on_epoch_complete(self, process_handler):
        while (text := self.read_stdout(process_handler)) is not None:
            if (index := text.find(self.ACC_STR)) != -1:
                test_str = text[index + len(self.ACC_STR) + 1:]
                if (acc_index := test_str.find("acc:")) != -1:
                    acc_str = test_str[acc_index + len("acc:") + 1:]
                    pct_index = acc_str.find("%")
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
        return f"models/pos_adv/pocketML/pos_{self.lang}_{self.treebank}"

    def model_path(self):
        return f"{self.model_base_path()}/pocketML"

    def predict_path(self):
        return f"{self.model_base_path()}/preds.out.task0"

class Stanford(Tagger):
    def __init__(self, args):
        super().__init__(args)
        with open(f"{self.model_base_path()}/pocketML.props", "w", encoding="utf-8") as fp:
            architecture = (
                "bidirectional5words,allwordshapes(-1,1)," +
                "rareExtractor(edu.stanford.nlp.tagger.maxent.ExtractorUCase)," +
                "rareExtractor(edu.stanford.nlp.tagger.maxent.ExtractorCNumber)," +
                "rareExtractor(edu.stanford.nlp.tagger.maxent.ExtractorDash)," +
                "rareExtractor(edu.stanford.nlp.tagger.maxent.ExtractorLetterDigitDash)," +
                "rareExtractor(edu.stanford.nlp.tagger.maxent.CompanyNameDetector)," +
                "rareExtractor(edu.stanford.nlp.tagger.maxent.ExtractorAllCapitalized)," +
                "rareExtractor(edu.stanford.nlp.tagger.maxent.ExtractorUpperDigitDash)," +
                "rareExtractor(edu.stanford.nlp.tagger.maxent.ExtractorStartSentenceCap)," +
                "rareExtractor(edu.stanford.nlp.tagger.maxent.ExtractorMidSentenceCapC)," +
                "rareExtractor(edu.stanford.nlp.tagger.maxent.ExtractorMidSentenceCap)," +
                "prefix(10),suffix(10),unicodeshapes(0),rareExtractor(" +
                "edu.stanford.nlp.tagger.maxent.ExtractorNonAlphanumeric)"
            )
            fp.write(f"arch = {architecture}\n")
            fp.write(f"model = {self.model_path()}\n")
            fp.write("encoding = UTF-8\n")
            fp.write(f"iterations = {args.iter}\n")
            fp.write(f"lang = {data_archives.LANGUAGES[args.lang]}\n")
            fp.write("tagSeparator = \\t\n")
            train_set = data_archives.get_dataset_path(self.lang, self.treebank, "train")
            fp.write(f"trainFile = format=TSV,wordColumn=0,tagColumn=1,{train_set}")

    async def on_epoch_complete(self, process_handler):
        while (text := self.read_stdout(process_handler)) is not None:
            if "Iter." in text:
                self.epoch += 1
                yield None

    def get_pred_acc(self):
        with open(self.predict_path(), "r", encoding="ansi") as fp:
            lines = fp.readlines()
            acc_str = lines[-2].split(None)[4]
            acc = float(acc_str[1:-3].replace(",", "."))
            return acc

    def model_base_path(self):
        return f"models/stanford-tagger/{self.lang}_{self.treebank}"

    def model_path(self):
        return f"{self.model_base_path()}/pocketML.tagger"

    def predict_path(self):
        return f"{self.model_base_path()}/preds.out"
