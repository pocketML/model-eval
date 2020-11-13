import nltk
from os import path
import dill as pickle
from util import data_archives
from taggers.tagger_wrapper import Tagger
from loadbar import Loadbar
from util.code_size import get_code_size

class ImportedTagger(Tagger):
    IS_IMPORTED = True

    def __init__(self, args, model_name, load_model=False):
        super().__init__(args, model_name)
        if load_model:
            if self.saved_model_exists():
                print(f"Loading saved '{model_name}' model.")
                self.load_model()
            else:
                print(f"Error: No trained model for '{self.model_name}' exists!")
    
    def evaluate(self, test_data, pipe):
        if self.args.loadbar:
            loadbar = Loadbar(50, len(test_data), f"Evaluating '{self.model_name}'")
            loadbar.print_bar()

        total = 0
        correct = 0
        curr_sent_correct = 0
        correct_sent = 0
        total_sent = 0
        prev_pct = -1
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
            if self.args.loadbar:
                loadbar.step()
            elif pct > prev_pct:
                print(f"{pct}%", end="\r", flush=True)
                prev_pct = pct
        pipe.send((correct / total, correct_sent / total_sent))

    def predict(self, sentence):
        return self.model.tag(sentence)

    def train(self, train_data):
        return self.model.train(train_data)

    def model_base_path(self):
        return f"models/{self.model_name}/{self.args.lang}_{self.args.treebank}"

    def model_path(self):
        return f"{self.model_base_path()}/model.pkl"

    def saved_model_exists(self):
        return path.exists(self.model_path())

    def save_model(self):
        with open(self.model_path(), "wb") as fp:
            pickle.dump(self.model, fp)

    def load_model(self):
        with open(self.model_path(), "rb") as fp:
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

    def code_size(self):
        return get_code_size(self.model.__class__.__module__)
