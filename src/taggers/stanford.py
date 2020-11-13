from taggers.tagger_wrapper_syscall import SysCallTagger
import os
import shutil
from util import data_archives

class Stanford(SysCallTagger):
    def __init__(self, args, model_name, load_model=False):
        super().__init__(args, model_name, load_model)

        if "JAVAHOME" not in os.environ:
            java_exe_path = shutil.which("java")
            if java_exe_path is None:
                print("WARNING: 'JAVAHOME' environment variable not set!")
            else:
                java_path = "/".join(java_exe_path.replace("\\", "/").split("/")[:-1])
                os.environ["JAVAHOME"] = java_path

        with open(f"{self.model_base_path()}/pocketML.props", "w", encoding="utf-8") as fp:
            architecture = (
                "bidirectional5words,allwordshapes(-1,1)," +
                "rareExtractor(edu.stanford.nlp.tagger.maxent.ExtractorUCase)," +
                "rareExtractor(edu.stanford.nlp.tagger.maxent.ExtractorCNumber)," +
                "rareExtractor(edu.stanford.nlp.tagger.maxent.ExtractorDash)," +
                "rareExtractor(edu.stanford.nlp.tagger.maxent.ExtractorLetterDigitDash)," +
                "rareExtractor(edu.stanford.nlp.tagger.maxent.CompanyNameDetector)," +
                "rareExtractor(edu.stanford.nlp.tagger.maxent.ExtractorAllCapitalized)," +
                "rareExtractor(edu.stanford.nlp.tagger.maxent.ExtractorUpperDigitDash)," +
                "rareExtractor(edu.stanford.nlp.tagger.maxent.ExtractorStartSentenceCap)," +
                "rareExtractor(edu.stanford.nlp.tagger.maxent.ExtractorMidSentenceCapC)," +
                "rareExtractor(edu.stanford.nlp.tagger.maxent.ExtractorMidSentenceCap)," +
                "prefix(10),suffix(10),unicodeshapes(0),rareExtractor(" +
                "edu.stanford.nlp.tagger.maxent.ExtractorNonAlphanumeric)"
            )
            fp.write(f"arch = {architecture}\n")
            fp.write(f"model = {self.model_path()}\n")
            fp.write("encoding = UTF-8\n")
            fp.write(f"iterations = {self.args.iter}\n")
            fp.write(f"lang = {data_archives.LANGUAGES[self.args.lang]}\n")
            fp.write("tagSeparator = \\t\n")
            train_set = data_archives.get_dataset_path(self.args.lang, self.args.treebank, "train")
            fp.write(f"trainFile = format=TSV,wordColumn=0,tagColumn=1,{train_set}")

    async def on_epoch_complete(self, process_handler):
        while (text := self.read_stdout(process_handler)) is not None:
            if "Iter." in text:
                self.epoch += 1
                yield None

    def evaluate(self):
        with open(self.predict_path(), "r", encoding="ansi") as fp:
            lines = fp.readlines()
            sent_acc_str = lines[-3].split(None)[4]
            sent_acc = float(sent_acc_str[1:-3].replace(",", "."))
            token_acc_str = lines[-2].split(None)[4]
            token_acc = float(token_acc_str[1:-3].replace(",", "."))
            return token_acc, sent_acc

    def model_base_path(self):
        return f"models/stanford/pocketML/{self.args.lang}_{self.args.treebank}"

    def model_path(self):
        return f"{self.model_base_path()}/pocketML.tagger"

    def predict_path(self):
        return f"{self.model_base_path()}/preds.out"

    def script_path(self):
        return "models/stanford/stanford-postagger.jar"

    def train_string(self):
        return (
            "java -cp [script_path] "
            "edu.stanford.nlp.tagger.maxent.MaxentTagger -props [model_base_path]/pocketML.props"
        )

    def predict_string(self):
        return ( 
            "java -mx2g -cp [script_path] "
            "edu.stanford.nlp.tagger.maxent.MaxentTagger -model [model_path] "
            "--encoding UTF-8 "
            "--testFile format=TSV,wordColumn=0,tagColumn=1,[dataset_test] [stdout] [pred_path]"
        )

    def code_size(self):
        return 0
