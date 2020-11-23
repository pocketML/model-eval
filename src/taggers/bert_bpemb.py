from os.path import getsize
from pathlib import Path
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
        bpemb_path = Path.home() / '.cache' / 'bpemb' / self.args.lang
        print(f'bpemb path: {bpemb_path}')
        embeddings_size = data_archives.get_folder_size(bpemb_path)
        total_size = PYTHON_STDLIB_SIZE + embeddings_size
        for glob_str in code_files:
            files = glob(glob_str)
            for file in files:
                total_size += getsize(file)
        return int(total_size)

    def model_size(self):
        filenames = glob(self.model_base_path() + '/**/*_model.pt', recursive=True)
        filenames.sort(key= lambda x: float(x.split('acc_')[1].split('_model')[0]))
        filename = filenames[-1]
        total_size = getsize(filename)
        
        # if using bert
        bert_path = Path.home() / '.cache' / 'torch' / 'transformers'
        total_size += data_archives.get_folder_size(bert_path)

        return int(total_size)
