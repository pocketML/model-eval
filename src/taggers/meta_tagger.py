from taggers.tagger_wrapper_syscall import SysCallTagger

class METATAGGER(SysCallTagger):
    ACC_STR = ['dev set', 'dev accuracies:']

    def __init__(self, args, model_name, load_model=False):
        super().__init__(args, model_name, load_model, simplified_dataset=False)

    async def on_epoch_complete(self, process_handler):

        while (text := self.read_stdout(process_handler)) is not None:
            if text.find('INFO:tensorflow')  != -1 and (index := text.find(self.ACC_STR[0])) != -1:
                test_str = text[index:].strip().split(' ')[4]
                self.epoch += 1
                yield float(test_str)
            elif text.find('INFO:tensorflow') != -1 and (index := text.find(self.ACC_STR[1])) != -1:
                test_str = text[index:].strip().split(' ')[3]
                self.epoch += 1
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
        return (
            'python [script_path] '
            '--train=[dataset_train] '
            '--dev=[dataset_dev] '
            f'--embeddings=[embeddings] '
            '--task=xtag '
            '--config=config.json '
            f'--output_dir=[model_base_path]'
        )
    
    def predict_string(self):
        return (
            'python models/meta_tagger/test_cw.py '
            '--test=[dataset_test] '
            '--task=xtag '
            f'--output_dir=[model_base_path] '
            '--out=[model_base_path]/preds.out'
        )