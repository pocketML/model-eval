import nltk
from os import path
import dill as pickle
import data_archives
from taggers.tagger_wrapper_syscall import Tagger

class ImportedTagger(Tagger):
    IS_IMPORTED = True
    MODEL_PATH = "models/nltk_pickled"

    def __init__(self, args, model_name, load_model=False):
        super().__init__(args, model_name)
        if load_model and self.saved_model_exists():
            self.load_model()

    def evaluate(self, test_data, pipe):
        total = 0
        correct = 0
        curr_sent_correct = 0
        correct_sent = 0
        total_sent = 0
        prev_pct = 0
        for index, sentence in enumerate(test_data):
            only_words = nltk.tag.untag(sentence)
            preds = self.predict(only_words)
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

    def predict(self, words):
        return self.model.tag(words)

    def train(self, train_data):
        return self.model.train(train_data)

    def saved_model_exists(self):
        return path.exists(f"{ImportedTagger.MODEL_PATH}/{self.model_name}.pk")

    def save_model(self):
        with open(f"{ImportedTagger.MODEL_PATH}/{self.model_name}.pk", "wb") as fp:
            pickle.dump(self.model, fp)

    def load_model(self):
        with open(f"{ImportedTagger.MODEL_PATH}/{self.model_name}.pk", "rb") as fp:
            self.model = pickle.load(fp)

    def format_data(self, dataset_type):
        data_path = data_archives.get_dataset_path(self.args.lang,
                                                   self.args.treebank,
                                                   dataset_type)
        train_data = open(data_path, "r", encoding="utf-8").readlines()
        sentences = []
        curr_senteces = []
        for line in train_data:
            if line.strip() == "":
                sentences.append(curr_senteces)
                curr_senteces = []
            else:
                curr_senteces.append(tuple(line.split(None)))
        return sentences
