from os.path import getsize
from pathlib import Path
from glob import glob
from os import path, walk
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
        import os
        os.environ["MKL_THREADING_LAYER"] = "GNU"
        return (
            'python [script_path_train] train '
            '--dataset ud_1_2 '
            '--lang [lang] '
            '--tag upostag '
            '--use-char '
            '--use-bpe '
            '--use-meta-rnn '
            '--use-bert ' # requires A LOT of mem
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
            'python [script_path_train] eval '
            '--dataset ud_1_2 '
            '--lang [lang] '
            '--tag upostag '
            '--use-char '
            '--use-bpe '
            '--use-meta-rnn '
            '--use-bert ' # requires A LOT of mem
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

    def code_size(self):
        base = "models/bert_bpemb"
        code_files = [
            f"{base}/*.py",
        ]
        total_size = PYTHON_STDLIB_SIZE
        for glob_str in code_files:
            files = glob(glob_str)
            for file in files:
                total_size += getsize(file)
        return int(total_size)

    def necessary_model_files(self):    
        model_paths = glob(self.model_base_path() + '/**/*_model.pt', recursive=True)
        model_paths.sort(key= lambda x: float(x.split('acc_')[1].split('_model')[0]))
        filenames = [model_paths[-1]]

        # if using bert
        bert_path = Path.home() / '.cache' / 'torch' / 'transformers'
        for dirpath, _, filenames in walk(bert_path):
            for f in filenames:
                filenames.append(path.join(dirpath, f))

        # Embeddings
        bpemb_path = Path.home() / '.cache' / 'bpemb' / self.args.lang
        for dirpath, _, filenames in walk(bpemb_path):
            for f in filenames:
                filenames.append(path.join(dirpath, f))

        return filenames
