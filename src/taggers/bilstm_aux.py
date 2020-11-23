from os.path import getsize
from glob import glob
from taggers.tagger_wrapper_syscall import SysCallTagger
from util.code_size import PYTHON_STDLIB_SIZE
from util import data_archives

class BILSTMAUX(SysCallTagger):
    ACC_STR = "dev accuracy:"

    def __init__(self, args, model_name, load_model=False):
        super().__init__(args, model_name, load_model)

    async def on_epoch_complete(self, process_handler):
        while (text := self.read_stdout(process_handler)) is not None:
            if (index := text.find(self.ACC_STR)) != -1:
                acc_str = text[index + len(self.ACC_STR) + 1:]
                self.epoch += 1
                yield float(acc_str)

    def evaluate(self):
        return super().evaluate(".task0")

    def model_base_path(self):
        return f"models/bilstm_aux/pocketML/{self.args.lang}_{self.args.treebank}"

    def model_path(self):
        return f"{self.model_base_path()}/pocketML"

    def predict_path(self):
        return f"{self.model_base_path()}/preds.out"

    def script_path_train(self):
        return "models/bilstm_aux/src/structbilty.py"

    def train_string(self):
        return (
            "python [script_path_train] --dynet-mem 1500 "
            "--train [dataset_train] --dev [dataset_dev] --test [dataset_test] "
            "--iters [iters] --model [model_path] --embeds [embeddings]"
        )

    def predict_string(self):
        return (
            "python [script_path_test] --model [model_path] "
            "--test [dataset_test] "
            "--output [pred_path]"
        )

    def code_size(self):
        depend_dynet_size = 5.1e6 # 5.1 MB
        depend_numpy_size = 13.0e6 # 13 MB
        depend_cython_size = 1.7e6 # 1.7 MB
        dependency_size = (
            depend_dynet_size + depend_numpy_size + depend_cython_size
        )
        base = "models/bilstm_aux/"
        code_files = [
            f"{base}/src/*.py",
            f"{base}/src/lib/*.py",
        ]
        embeddings_size = data_archives.get_embeddings_size(self.args.lang)
        total_size = PYTHON_STDLIB_SIZE + embeddings_size + dependency_size
        for glob_str in code_files:
            files = glob(glob_str)
            for file in files:
                total_size += getsize(file)
        return int(total_size)

    def necessary_model_files(self):
        return [
            self.model_path() + ".model",
            self.model_path() + ".params.pickle"
        ]
