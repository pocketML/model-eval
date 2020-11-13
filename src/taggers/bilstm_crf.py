from glob import glob
from taggers.tagger_wrapper_syscall import SysCallTagger

class BILSTMCRF(SysCallTagger):
    ACC_STR = "test loss:"

    def __init__(self, args, model_name, load_model=False):
        super().__init__(args, model_name, load_model,  simplified_dataset=False)

    async def on_epoch_complete(self, process_handler):
        while (text := self.read_stdout(process_handler)) is not None:
            if (index := text.find(self.ACC_STR)) != -1:
                test_str = text[index + len(self.ACC_STR) + 1:]
                if (acc_index := test_str.find("acc:")) != -1:
                    acc_str = test_str[acc_index + len("acc:") + 1:]
                    pct_index = acc_str.find("%")
                    self.epoch += 1
                    yield float(acc_str[:pct_index])

    def model_base_path(self):
        return f"models/bilstm_crf/pocketML/pos_{self.args.lang}_{self.args.treebank}"

    def model_path(self):
        folders = glob(f"{self.model_base_path()}/save/epoch*")
        if len(folders) == 0:
            return ''
        folders.sort(key=lambda x: int(x.replace("\\", "/").split("/")[-1][5:]))
        return folders[-1].replace("\\", "/") + "/final.npz"

    def predict_path(self):
        return f"{self.model_base_path()}/preds.out"

    def script_path_train(self):
        return "models/bilstm_crf/bilstm_bilstm_crf.py"

    def train_string(self):
        import os
        os.environ['THEANO_FLAGS'] = "mode=FAST_RUN,device=cpu,floatX=float32,blas.ldflags=''"
        return (
            "python [script_path_train] --fine_tune --embedding polyglot --oov embedding --update momentum --adv 0.05 "
            "--batch_size 10 --num_units 150 --num_filters 50 --learning_rate 0.01 --decay_rate 0.05 --grad_clipping 5 --regular none --dropout "
            "--train [dataset_train] "
            "--dev [dataset_dev] "
            "--test [dataset_test] "
            "--embedding_dict [embeddings] "
            "--patience 30 --exp_dir [model_base_path] [reload]"
        )

    def predict_string(self):
        import os
        os.environ['THEANO_FLAGS'] = "mode=FAST_RUN,device=cpu1,floatX=float32,blas.ldflags=''"
        return self.train_string() + " --output_prediction"

    def reload_string(self):
        return f"--reload {self.model_path()}"

    def code_size(self):
        return 0
