from abc import ABC, abstractmethod
from os import path
from glob import glob
import nltk
from nltk.tbl.template import Template
from nltk.tag.brill import Word, Pos
import dill as pickle
import data_archives

class Tagger(ABC):
    def __init__(self, args, model_name):
        self.args = args
        self.model_name = model_name

class SysCallTagger(Tagger):
    def __init__(self, args, model_name):
        super().__init__(args, model_name)
        self.epoch = 0
        self.model_name = model_name

    def read_stdout(self, process_handler):
        if process_handler.poll() is not None:
            return None
        data = process_handler.stdout.readline()
        text = data.decode("utf-8")
        return text

    def inference_complete(self, process_handler):
        return process_handler.poll() is not None

    def evaluate(self, ext=""):
        total = 0
        correct = 0
        curr_sent_correct = 0
        curr_sent_count = 0
        correct_sent = 0
        total_sent = 0

        with open(self.predict_path() + ext, "r", encoding="utf-8") as fp:
            lines = fp.readlines()
            for line in lines:
                line = line.strip()
                if line == "":
                    total_sent += 1
                    if curr_sent_count == curr_sent_correct:
                        correct_sent += 1
                    curr_sent_correct = 0
                    curr_sent_count = 0
                    continue
                total += 1
                curr_sent_count += 1
                split = line.split(None)
                predicted = split[1]
                actual = split[2]
                if predicted == actual:
                    correct += 1
                    curr_sent_correct += 1

        return correct / total, correct_sent / total_sent

    @abstractmethod
    def model_base_path(self):
        pass

    @abstractmethod
    def model_path(self):
        pass

    @abstractmethod
    def predict_path(self):
        pass

class SVMT(SysCallTagger):
    ACC_STR = "TEST ACCURACY:"

    def __init__(self, args, model_name):
        super().__init__(args, model_name)
        cfg_path = "models/svmtool/bin/config.svmt"
        with open(cfg_path, "r", encoding="utf-8", newline=None) as fp:
            lines = fp.readlines()
        train_set = data_archives.get_dataset_path(self.args.lang, self.args.treebank, "train")
        test_set = data_archives.get_dataset_path(self.args.lang, self.args.treebank, "test")
        dev_set = data_archives.get_dataset_path(self.args.lang, self.args.treebank, "dev")
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

    def model_base_path(self):
        return f"models/svmtool/pocketML/{self.args.lang}_{self.args.treebank}"

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

class BILSTM(SysCallTagger):
    ACC_STR = "dev accuracy:"

    async def on_epoch_complete(self, process_handler):
        while (text := self.read_stdout(process_handler)) is not None:
            if (index := text.find(self.ACC_STR)) != -1:
                acc_str = text[index + len(self.ACC_STR) + 1:]
                self.epoch += 1
                yield float(acc_str)

    def get_pred_acc(self):
        return super().evaluate(".task0")

    def model_base_path(self):
        return f"models/bilstm-aux/pocketML/{self.args.lang}_{self.args.treebank}"

    def model_path(self):
        return f"{self.model_base_path()}/pocketML"

    def predict_path(self):
        return f"{self.model_base_path()}/preds.out"

class POSADV(SysCallTagger):
    ACC_STR = "test loss:"

    async def on_epoch_complete(self, process_handler):
        while (text := self.read_stdout(process_handler)) is not None:
            print(text)
            if (index := text.find(self.ACC_STR)) != -1:
                test_str = text[index + len(self.ACC_STR) + 1:]
                if (acc_index := test_str.find("acc:")) != -1:
                    acc_str = test_str[acc_index + len("acc:") + 1:]
                    pct_index = acc_str.find("%")
                    self.epoch += 1
                    yield float(acc_str[:pct_index])

    def model_base_path(self):
        return f"models/pos_adv/pocketML/pos_{self.args.lang}_{self.args.treebank}"

    def model_path(self):
        return f"{self.model_base_path()}/pocketML"

    def predict_path(self):
        predict_files = glob(f"{self.model_base_path()}/eval/test*")
        if len(predict_files) == 0:
            return self.model_base_path()
        predict_files.sort(key=lambda x: int(x.replace("\\", "/").split("/")[-1]))
        return predict_files[-1]

class Stanford(SysCallTagger):
    def __init__(self, args, model_name):
        super().__init__(args, model_name)
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
            fp.write(f"iterations = {self.args.iter}\n")
            fp.write(f"lang = {data_archives.LANGUAGES[args.args.lang]}\n")
            fp.write("tagSeparator = \\t\n")
            train_set = data_archives.get_dataset_path(self.args.lang, self.args.treebank, "train")
            fp.write(f"trainFile = format=TSV,wordColumn=0,tagColumn=1,{train_set}")

    async def on_epoch_complete(self, process_handler):
        while (text := self.read_stdout(process_handler)) is not None:
            if "Iter." in text:
                self.epoch += 1
                yield None

    def evaluate(self):
        with open(self.predict_path(), "r", encoding="ansi") as fp:
            lines = fp.readlines()
            sent_acc_str = lines[-3].split(None)[4]
            sent_acc = float(sent_acc_str[1:-3].replace(",", "."))
            token_acc_str = lines[-2].split(None)[4]
            token_acc = float(token_acc_str[1:-3].replace(",", "."))
            return token_acc, sent_acc

    def model_base_path(self):
        return f"models/stanford-tagger/{self.args.lang}_{self.args.treebank}"

    def model_path(self):
        return f"{self.model_base_path()}/pocketML.tagger"

    def predict_path(self):
        return f"{self.model_base_path()}/preds.out"

class NLTKTagger(Tagger):
    MODEL_PATH = "models/nltk_pickled"

    def __init__(self, args, model_name, load_model=False):
        super().__init__(args, model_name)
        if load_model and self.saved_model_exists():
            self.nltk_model = self.load_model()

    def evaluate(self, test_data, pipe):
        total = 0
        correct = 0
        curr_sent_correct = 0
        correct_sent = 0
        total_sent = 0
        prev_pct = 0
        for index, sentence in enumerate(test_data):
            only_words = nltk.tag.untag(sentence)
            preds = self.nltk_model.tag(only_words)
            curr_sent_correct = 0
            for test_tup, pred_tup in zip(sentence, preds):
                if test_tup[1] == pred_tup[1]:
                    correct += 1
                    curr_sent_correct += 1
                total += 1
            if curr_sent_correct == len(sentence):
                correct_sent += 1
            total_sent += 1

            pct = int((index / len(test_data)) * 100)
            if pct > prev_pct:
                prev_pct = pct
                print(f"{pct}%", end="\r", flush=True)
        pipe.send((correct / total, correct_sent / total_sent))

    def train(self, train_data):
        return self.nltk_model.train(train_data)

    def saved_model_exists(self):
        return path.exists(f"{NLTKTagger.MODEL_PATH}/{self.model_name}.pk")

    def save_model(self):
        with open(f"{NLTKTagger.MODEL_PATH}/{self.model_name}.pk", "wb") as fp:
            pickle.dump(self.nltk_model, fp)

    def load_model(self):
        with open(f"{NLTKTagger.MODEL_PATH}/{self.model_name}.pk", "rb") as fp:
            return pickle.load(fp)

class TnT(NLTKTagger):
    def __init__(self, args, model_name, load_model=False):
        super().__init__(model_name, args, load_model)
        if not load_model:
            self.nltk_model = nltk.TnT()

class Brill(NLTKTagger):
    def __init__(self, args, model_name, load_model=False):
        super().__init__(args, model_name, load_model)
        if not load_model:
            base_tagger = TnT("tnt", args, load_model=True)
            features = [
                Template(Pos([-1])), Template(Pos([1])), Template(Pos([-2])),
                Template(Pos([2])), Template(Word([0])), Template(Word([1, -1]))
            ]
            self.nltk_model = nltk.BrillTaggerTrainer(base_tagger.nltk_model, features)

class CRF(NLTKTagger):
    def __init__(self, args, model_name, load_model=False):
        super().__init__(args, model_name, load_model)
        if not load_model:
            features = self.word_features
            train_opts = {
                "c1": 1.0,
                "c2": 1e-3,
                "max_iterations": self.args.iter,
                "feature.possible_transitions": True
            }
            self.nltk_model = nltk.CRFTagger(features, False, train_opts)

    def train(self, train_data):
        self.nltk_model.train(train_data, f"{NLTKTagger.MODEL_PATH}/{self.model_name}.crfsuite")

    def save_model(self):
        pass

    def word_features(self, sentence, i):
        word = sentence[i]
        features = [
            'bias',
            'word.lower=' + word.lower(),
            'word[-3:]=' + word[-3:],
            'word[-2:]=' + word[-2:],
            'word.isupper=%s' % word.isupper(),
            'word.istitle=%s' % word.istitle(),
            'word.isdigit=%s' % word.isdigit()
        ]
        if i > 0:
            word1 = sentence[i-1]
            features.extend([
                '-1:word.lower=' + word1.lower(),
                '-1:word.istitle=%s' % word1.istitle(),
                '-1:word.isupper=%s' % word1.isupper()
            ])
        else:
            features.append('BOS')

        if i < len(sentence)-1:
            word1 = sentence[i+1]
            features.extend([
                '+1:word.lower=' + word1.lower(),
                '+1:word.istitle=%s' % word1.istitle(),
                '+1:word.isupper=%s' % word1.isupper()
            ])
        else:
            features.append('EOS')

        return features
