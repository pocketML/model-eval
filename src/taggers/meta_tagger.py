from taggers.tagger_wrapper_syscall import SysCallTagger
import os

class METATAGGER(SysCallTagger):
    ACC_STR = 'dev accuracies:'

    def __init__(self, args, model_name, load_model=False):
        super().__init__(args, model_name, load_model)

    async def on_epoch_complete(self, process_handler):
        while (text := self.read_stdout(process_handler)) is not None:
            if (text.find(self.ACC_STR)) != -1:
                test_str = text[len(self.ACC_STR)].strip().split(' ')[1]
                self.epoch += 5
                yield float(test_str)

    def model_base_path(self):
        return f'models/meta_tagger/pocketML/{self.args.lang}_{self.args.treebank}'
    
    def model_path(self):
        return ''

    def predict_path(self):
        return f'{self.model_base_path()}/preds.out'

    def script_path(self):
        return 'models/meta_tagger/train_cw.py'

    def train_string(self):
        cwd = os.getcwd().replace('\\', '/')
        return (
            'python [script_path] '
            '--train=[dataset_train] '
            '--dev=\'[dataset_dev]\' '
            f'--embeddings=[embeddings] '
            '--task=xtag '
            '--config=config.json '
            f'--output_dir=[model_base_path]'
        )
    
    def predict_string(self):
        return (
            'python models/meta_tagger/test_cw.py '
            '--test=[dataset_test] '
            f'--output_dir=[model_base_path] '
            '--out=preds.out'
        )