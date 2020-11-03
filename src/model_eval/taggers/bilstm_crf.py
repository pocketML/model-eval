from taggers.tagger_wrapper_syscall import SysCallTagger
from glob import glob

class BILSTMCRF(SysCallTagger):
    ACC_STR = "test loss:"

    async def on_epoch_complete(self, process_handler):
        while (text := self.read_stdout(process_handler)) is not None:
            print(text)
            if (index := text.find(self.ACC_STR)) != -1:
                test_str = text[index + len(self.ACC_STR) + 1:]
                if (acc_index := test_str.find("acc:")) != -1:
                    acc_str = test_str[acc_index + len("acc:") + 1:]
                    pct_index = acc_str.find("%")
                    self.epoch += 1
                    yield float(acc_str[:pct_index])

    def model_base_path(self):
        return f"models/bilstm-crf/pocketML/pos_{self.args.lang}_{self.args.treebank}"

    def model_path(self):
        return f"{self.model_base_path()}/pocketML"

    def predict_path(self):
        predict_files = glob(f"{self.model_base_path()}/eval/test*")
        if len(predict_files) == 0:
            return self.model_base_path()
        predict_files.sort(key=lambda x: int(x.replace("\\", "/").split("/")[-1]))
        return predict_files[-1]

    def train_string(self):
        return (
            "python [dir]/models/bilstm-crf/bilstm_bilstm_crf.py --fine_tune --embedding polyglot --oov embedding --update momentum --adv 0.05 "
            "--batch_size 10 --num_units 150 --num_filters 50 --learning_rate 0.01 --decay_rate 0.05 --grad_clipping 5 --regular none --dropout "
            "--train [dataset_train] "
            "--dev [dataset_dev] "
            "--test [dataset_test] "
            "--embedding_dict [dir]/models/bilstm-crf/dataset/word_vec/polyglot-[lang].pkl "
            f"--output_prediction --patience 30 --exp_dir [model_base_path]"
            )

    def eval_string(self):
        None
