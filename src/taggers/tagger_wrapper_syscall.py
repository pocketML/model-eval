from abc import abstractmethod
from taggers.tagger_wrapper import Tagger

class SysCallTagger(Tagger):
    """
    This is the base class for all taggers that reside in external files/programs.
    This includes:
    bilstm-plank, bilstm-yasanuga, svmtool, stanford-tagger, meta-bilstm, and bert-bpemb.
    """
    IS_IMPORTED = False

    def __init__(self, args, model_name, load_model=False, simplified_dataset=True, simplified_eos_dataset=False):
        super().__init__(args, model_name, simplified_dataset=simplified_dataset, simplified_eos_dataset=simplified_eos_dataset)
        self.epoch = 0

    def read_stdout(self, process_handler):
        """
        Read training progress by parsing stdout, from a given subprocess.
        """
        if process_handler.poll() is not None:
            return None # Training process has terminated.
        data = process_handler.stdout.readline()
        text = data.decode("utf-8")
        if self.args.verbose and text.strip() != '':
            print(text)
        return text

    def is_inference_complete(self, process_handler):
        return process_handler.poll() is not None

    def evaluate(self, ext=""):
        """
        Before this method is called, the given tagger being evaluated will have outputted its
        predictions to a file. This method then runs through that file, and outputs 
        sentence and token accuracy of the tagger's predictions.
        Individual taggers can override this method, if the predictions they output are
        formatted differently, than what this method expects.
        """
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
                predicted = split[-1]
                actual = split[-2]
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
