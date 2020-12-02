from os.path import getsize
from glob import glob
from taggers.tagger_wrapper_syscall import SysCallTagger
from util.code_size import PYTHON_STDLIB_SIZE
from util import data_archives

class METATAGGER(SysCallTagger):
    ACC_STR = ['dev set', 'dev accuracies:']

    def __init__(self, args, model_name, load_model=False):
        super().__init__(args, model_name, load_model, simplified_dataset=False)

    async def on_epoch_complete(self, process_handler):
        while (text := self.read_stdout(process_handler)) is not None:
            if text.find('INFO:tensorflow') != -1 and (index := text.find(self.ACC_STR[0])) != -1:
                test_str = text[index:].strip().split(' ')[4]
                self.epoch += 1
                yield float(test_str)
            elif text.find('INFO:tensorflow') != -1 and (index := text.find(self.ACC_STR[1])) != -1:
                test_str = text[index:].strip().split(' ')[3]
                self.epoch += 1
                yield float(test_str)

    def evaluate(self, ext=""):
        total = 0
        correct = 0
        curr_sent_correct = 0
        curr_sent_count = 0
        correct_sent = 0
        total_sent = 0

        with open(self.predict_path() + ext, "r", encoding="utf-8") as fp_pred:
            with open(data_archives.get_dataset_path(self.args.lang, self.args.treebank, "test", False), encoding="utf-8") as fp_test:
                lines_pred = fp_pred.readlines()
                lines_data = fp_test.readlines()
                index_pred = 0
                index_data = 0
                while index_pred < len(lines_pred):
                    line_pred = lines_pred[index_pred].strip()
                    line_data = lines_data[index_data].strip()

                    if line_pred == "":
                        total_sent += 1
                        if curr_sent_count == curr_sent_correct:
                            correct_sent += 1
                        curr_sent_correct = 0
                        curr_sent_count = 0
                    else:
                        while line_data[0] == "#":
                            index_data += 1
                            line_data = lines_data[index_data]
                        total += 1
                        curr_sent_count += 1
                        split_pred = line_pred.split(None)
                        split_data = line_data.split(None)
                        predicted = split_pred[3]
                        actual = split_data[3]
                        if predicted == actual:
                            correct += 1
                            curr_sent_correct += 1

                    index_data += 1
                    index_pred += 1

        token_acc = correct / total if total > 0 else 0
        sent_acc = correct_sent / total_sent if total_sent > 0 else 0
        return token_acc, sent_acc

    def model_base_path(self):
        return f'models/meta_tagger/pocketML/{self.args.lang}_{self.args.treebank}'

    def model_path(self):
        glob_paths = glob(f"{self.model_base_path()}/*.data*")
        return glob_paths[0] if len(glob_paths) > 0 else ""

    def predict_path(self):
        return f'{self.model_base_path()}/preds.out'

    def script_path_train(self):
        return 'models/meta_tagger/train_cw.py'

    def script_path_test(self):
        return 'models/meta_tagger/test_cw.py'

    def train_string(self):
        return (
            'python -X utf8 [script_path_train] '
            '--train=[dataset_train] '
            '--dev=[dataset_dev] '
            f'--embeddings=[embeddings] '
            '--task=upos '
            '--config=config.json '
            f'--output_dir=[model_base_path]/'
        )

    def predict_string(self):
        return (
            'python -X utf8 [script_path_test] '
            '--test=[dataset_test] '
            '--task=upos '
            '--output_dir=[model_base_path]/ '
            '--out=[model_base_path]/preds.out'
        )

    def code_size(self):
        depend_tensorflow_size = 373629600 # 373 MB
        base = "models/meta_tagger"
        code_files = [
            f"{base}/*.py",
        ]
        embeddings_size = data_archives.get_embeddings_size(self.args.lang)
        total_size = PYTHON_STDLIB_SIZE + embeddings_size + depend_tensorflow_size
        for glob_str in code_files:
            files = glob(glob_str)
            for file in files:
                total_size += getsize(file)
        return int(total_size)

    def necessary_model_files(self):
        return [
            self.model_path(),
            self.model_base_path() + "/meta_word_char_v1.meta",
            data_archives.get_embeddings_path(self.args.lang)
        ]
