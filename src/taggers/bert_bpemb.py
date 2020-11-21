from os.path import getsize
from glob import glob
from taggers.tagger_wrapper_syscall import SysCallTagger
from util.code_size import PYTHON_STDLIB_SIZE
from util import data_archives

class BERT_BPEMB(SysCallTagger):
    ACC_STR = ['score acc_']

    def __init__(self, args, model_name, load_model=False):
        super().__init__(args, model_name, load_model, simplified_dataset=False)

    async def on_epoch_complete(self, process_handler):
        while (text := self.read_stdout(process_handler)) is not None:
            if (index := text.find(self.ACC_STR[0])) != -1:
                test_str = text[index:].strip().split('_')[1].split('/')[0]
                self.epoch += 1
                yield float(test_str)

    def evaluate(self, ext=""):
        '''total = 0
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
        return token_acc, sent_acc'''
        return 0.0, 0.0

    def model_base_path(self):
        return f'models/bert_bpemb/pocketML/{self.args.lang}_{self.args.treebank}'
    
    def model_path(self):
        return ''

    def predict_path(self):
        return f'{self.model_base_path()}/preds.out'

    def script_path_train(self):
        return 'models/bert_bpemb/main.py'

    def script_path_test(self):
        return self.script_path_train()

    def train_string(self):
        return (
            'python [script_path_train] train '
            '--dataset ud_1_2 '
            '--lang [lang] '
            '--tag upostag '
            '--use-char '
            '--use-bpe '
            '--use-meta-rnn '
            #'--use-bert '
            '--best-vocab-size '
            '--char-emb-dim 50 '
            '--char-nhidden 256 '
            '--bpe-nhidden 256 '
            '--meta-nhidden 256 '
            '--dropout 0.2 '
            '--data-dir [dataset_folder] '
            '--outdir [model_base_path] '
            f'--relative_path models/bert_bpemb'
        )

    def predict_string(self):
        return (
            'python [script_path_test] '
            '--test=[dataset_test] '
            '--task=upos '
            '--output_dir=[model_base_path]/ '
            '--out=[model_base_path]/preds.out'
        )

    def code_size(self):
        base = "models/bert_bpemb"
        '''code_files = [
            f"{base}/*.py",
        ]
        embeddings_size = data_archives.get_embeddings_size(self.args.lang)
        total_size = PYTHON_STDLIB_SIZE + embeddings_size + depend_tensorflow_size
        for glob_str in code_files:
            files = glob(glob_str)
            for file in files:
                total_size += getsize(file)
        return int(total_size)'''
        return 0

    def model_size(self):
        files = glob(f"{self.model_base_path()}/*")
        total_size = 0
        for file in files:
            total_size += getsize(file)
        return int(total_size)
