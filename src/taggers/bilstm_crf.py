from os.path import getsize
from glob import glob
from taggers.tagger_wrapper_syscall import SysCallTagger
from util.code_size import PYTHON_STDLIB_SIZE
from util import data_archives

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
        depend_theano_size = 16.2e6 # 16.2 MB
        depend_six_size = 13.0e3 # 10 KB
        depend_scipy_size = 31.4e6 # 31.4 MB
        depend_lasagne_size = 949.0e3 # 949 KB
        depend_smartopen_size = 187.0e3 # 187 KB
        depend_gensim_size = 24.2e6 # 24.2 MB
        dependency_size = (
            depend_theano_size + depend_six_size + depend_scipy_size +
            depend_lasagne_size + depend_smartopen_size + depend_gensim_size
        )
        base = "models/bilstm_crf/"
        code_files = [
            f"{base}/bilstm_bilstm_crf.py",
            f"{base}/lasagne_nlp/*/*.py",
        ]
        embeddings_size = data_archives.get_embeddings_size(self.args.lang)
        total_size = PYTHON_STDLIB_SIZE + embeddings_size + dependency_size
        for glob_str in code_files:
            files = glob(glob_str)
            for file in files:
                total_size += getsize(file)
        return int(total_size)
