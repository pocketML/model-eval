import nltk
from os import path
import dill as pickle
from util import data_archives
from taggers.tagger_wrapper import Tagger
from loadbar import Loadbar
from util.code_size import get_code_size

class ImportedTagger(Tagger):
    """
    This is the base class for all taggers that are imported into Python.
    This includes all NLTK models and Flair.
    """
    IS_IMPORTED = True

    def __init__(self, args, model_name, load_model=False):
        super().__init__(args, model_name)
        if load_model:
            if self.saved_model_exists():
                print(f"Loading saved '{model_name}' model from '{self.model_path()}'.")
                self.load_model()
            else:
                print(f"Error: No trained model for '{self.model_name}' exists!")
    
    def evaluate(self, test_data, pipe):
        """
        Run through all words/tags in sentences in 'test_data' and evaluate
        accuracy compared to the predictions of whatever tagger virtually calls this method.
        Since this method is run in a seperate process (to monitor memory usage),
        results for token and sentence accuracy is sent back to the main process through the 'pipe'.
        """
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

        token_acc = correct / total if total > 0 else 0
        sent_acc = correct_sent / total_sent if total_sent > 0 else 0
        pipe.send((token_acc, sent_acc))

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
        """
        Read and format a dataset of 'dataset_type' into a list
        of sentences of tuples of (word, tag) pairs.
        """
        data_path = data_archives.get_dataset_path(self.args.lang,
                                                   self.args.treebank,
                                                   dataset_type)
        train_data = open(data_path, "r", encoding="utf-8").readlines()
        sentences = []
        curr_sentences = []
        for line in train_data:
            stripped = line.strip()
            if stripped == "":
                sentences.append(curr_sentences)
                curr_sentences = []
            else:
                curr_sentences.append(tuple(stripped.split("\t")))
        return sentences

    def code_size(self):
        return int(get_code_size(self.model.__class__.__module__)["total_size"])
