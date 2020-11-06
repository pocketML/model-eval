from abc import abstractmethod
from taggers.tagger_wrapper import Tagger
import data_archives

class SysCallTagger(Tagger):
    IS_IMPORTED = False

    def __init__(self, args, model_name, load_model=False):
        super().__init__(args, model_name)
        self.epoch = 0

    def read_stdout(self, process_handler):
        if process_handler.poll() is not None:
            return None
        data = process_handler.stdout.readline()
        text = data.decode("utf-8")
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

        return correct / total, correct_sent / total_sent

    def reload_string(self):
        return None

    def embeddings_path(self):
        return data_archives.get_embeddings_path(self.args.lang)

    @abstractmethod
    def model_base_path(self):
        pass

    @abstractmethod
    def model_path(self):
        pass

    @abstractmethod
    def predict_path(self):
        pass

    @abstractmethod
    def train_string(self):
        pass

    @abstractmethod
    def predict_string(self):
        pass
