from taggers.tagger_wrapper_syscall import SysCallTagger

class BILSTMAUX(SysCallTagger):
    ACC_STR = "dev accuracy:"

    async def on_epoch_complete(self, process_handler):
        while (text := self.read_stdout(process_handler)) is not None:
            print(text)
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

    def train_string(self):
        return (
            "python [dir]/models/bilstm_aux/src/structbilty.py --dynet-mem 1500 "
            "--train [dataset_train] --dev [dataset_dev] --test [dataset_test] "
            "--iters [iters] --model [model_path]" #--embeds [embeddings] --embeds_in_file"
        )

    def predict_string(self):
        return (
            "python [dir]/models/bilstm_aux/src/structbilty.py --model [model_path] "
            "--test [dataset_test] "
            "--output [pred_path]"
        )
