from os.path import getsize
from glob import glob
from taggers.tagger_wrapper_syscall import SysCallTagger
from util import data_archives
from util.code_size import PERL_STDLIB_SIZE

class SVMT(SysCallTagger):
    ACC_STR = "TEST ACCURACY:"

    def __init__(self, args, model_name, load_model=False):
        super().__init__(args, model_name, load_model, simplified_eos_dataset=True)
        cfg_path = "models/svmtool/bin/config.svmt"
        with open(cfg_path, "r", encoding="utf-8", newline=None) as fp:
            lines = fp.readlines()
        train_set = data_archives.get_dataset_path(self.args.lang, self.args.treebank, "train", simplified=False, eos=True)
        test_set = data_archives.get_dataset_path(self.args.lang, self.args.treebank, "test", simplified=False, eos=True)
        dev_set = data_archives.get_dataset_path(self.args.lang, self.args.treebank, "dev", simplified=False, eos=True)
        lines[3] = f"TRAINSET = {train_set}\n"
        lines[5] = f"VALSET = {dev_set}\n"
        lines[7] = f"TESTSET = {test_set}\n"
        lines[11] = f"NAME = {self.model_base_path()}/pocketML\n"
        with open(cfg_path, "w", encoding="utf-8", newline="\n") as fp:
            for line in lines:
                fp.write(line)

    async def on_epoch_complete(self, process_handler):
        while (text := self.read_stdout(process_handler)) is not None:
            if (index := text.find(self.ACC_STR)) != -1:
                acc_str = text[index + len(self.ACC_STR) + 1:]
                if (pct_index := acc_str.find("%")) != -1:
                    self.epoch += 1
                    yield float(acc_str[:pct_index])

    def model_base_path(self):
        return f"models/svmtool/pocketML/{self.args.lang}_{self.args.treebank}"

    def model_path(self):
        files = glob(f"{self.model_base_path()}/pocketML.FLD.*")
        if len(files) == 0:
            return self.model_base_path()
        split_files = [x.split(".") for x in files]
        
        split_files = [int(x[x.index("FLD") + 1]) for x in split_files]
        split_files.sort()
        return f"{self.model_base_path()}/pocketML.FLD.{split_files[-1]}"

    def predict_path(self):
        return f"{self.model_base_path()}/preds.out"

    def script_path_train(self):
        return "models/svmtool/bin/SVMTlearn.pl"

    def script_path_test(self):
        return "models/svmtool/bin/SVMTagger.pl"

    def train_string(self):
        return (
            "bash -c \"perl [script_path_train] -V 1 models/svmtool/bin/config.svmt\""
        )

    def predict_string(self):
        return (
            f"bash -c \"perl [script_path_test] [model_path] < "
            f"[dataset_test] > [pred_path]\""
        )

    def code_size(self):
        base = "models/svmtool/"
        code_files = [
            f"bin/*.pl",
            f"lib/SVMTool/*.pm",
            f"svmlight/*"
        ]
        total_size = PERL_STDLIB_SIZE
        for glob_str in code_files:
            files = glob(f"{base}/{glob_str}")
            for file in files:
                total_size += getsize(file)
        return int(total_size)

    def necessary_model_files(self):
        base = self.model_path()
        exclude_files = [
            "*.SAMPLES*",
            "*.W",
            "*.TEST",
            "*.TRAIN"
        ]
        glob_all_files = set(glob(f"{base}*"))
        glob_exclude_files = set()
        for glob_exclude in exclude_files:
            for filename in glob(f"{base}{glob_exclude}"):
                glob_exclude_files.add(filename)
        
        return list(glob_all_files - glob_exclude_files)

    def evaluate(self, ext=""):
        """
        Before this method is called, the given tagger being evaluated will have outputted its
        predictions to a file. This method then runs through that file, and outputs 
        sentence and token accuracy of the tagger's predictions.
        Individual taggers can override this method, if the predictions they output are
        formatted differently, than what this method expects.
        """
        total = 0
        correct = 0
        curr_sent_correct = 0
        curr_sent_count = 0
        correct_sent = 0
        total_sent = 0

        test_data = data_archives.get_dataset_path(self.args.lang, self.args.treebank, "test", simplified=self.simplified_dataset, eos=self.simplified_eos_dataset)

        print(test_data)
        #exit()

        with open(self.predict_path() + ext, "r", encoding="utf-8") as fp:
            lines = fp.readlines()
            for line in lines:
                line = line.strip()
                if line == "":
                    total_sent += 1
                    if curr_sent_count == curr_sent_correct:
                        correct_sent += 1
                    curr_sent_correct = 0
                    curr_sent_count = 0
                    continue
                total += 1
                curr_sent_count += 1
                split = line.split(None)
                predicted = split[-1]
                actual = split[-2]
                if predicted == actual:
                    correct += 1
                    curr_sent_correct += 1

        token_acc = correct / total if total > 0 else 0
        sent_acc = correct_sent / total_sent if total_sent > 0 else 0
        return token_acc, sent_acc