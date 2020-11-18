from abc import abstractmethod
from os.path import getsize
from taggers.tagger_wrapper import Tagger

class SysCallTagger(Tagger):
    IS_IMPORTED = False

    def __init__(self, args, model_name, load_model=False, simplified_dataset=True):
        super().__init__(args, model_name, simplified_dataset=simplified_dataset)
        self.epoch = 0

    def read_stdout(self, process_handler):
        if process_handler.poll() is not None:
            return None
        data = process_handler.stdout.readline()
        text = data.decode("utf-8")
        print(text)
        return text

    def is_inference_complete(self, process_handler):
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

        token_acc = correct / total if total > 0 else 0
        sent_acc = correct_sent / total_sent if total_sent > 0 else 0
        return token_acc, sent_acc

    def reload_string(self):
        return None

    @abstractmethod
    def script_path_train(self):
        pass

    def script_path_test(self):
        return self.script_path_train()

    @abstractmethod
    def predict_path(self):
        pass

    @abstractmethod
    def train_string(self):
        pass

    @abstractmethod
    def predict_string(self):
        pass

    def model_size(self):
        return getsize(self.model_path())
