import os
import shutil
from glob import glob
from taggers.tagger_wrapper_syscall import SysCallTagger
from util import data_archives
from util.code_size import JAVA_JRE_SIZE

PROPS_ENGLISH = { # Copied from stanfords example model english-left3words-distsim.
    "arch": (
        "bidirectional5words,wordshapes(-1,1),"
        ",rareExtractor(edu.stanford.nlp.tagger.maxent.ExtractorUCase),"
        "rareExtractor(edu.stanford.nlp.tagger.maxent.ExtractorCNumber),"
        "rareExtractor(edu.stanford.nlp.tagger.maxent.ExtractorDash),"
        "rareExtractor(edu.stanford.nlp.tagger.maxent.ExtractorLetterDigitDash),"
        "rareExtractor(edu.stanford.nlp.tagger.maxent.CompanyNameDetector),"
        "rareExtractor(edu.stanford.nlp.tagger.maxent.ExtractorAllCapitalized),"
        "rareExtractor(edu.stanford.nlp.tagger.maxent.ExtractorUpperDigitDash),"
        "rareExtractor(edu.stanford.nlp.tagger.maxent.ExtractorStartSentenceCap),"
        "rareExtractor(edu.stanford.nlp.tagger.maxent.ExtractorMidSentenceCapC),"
        "rareExtractor(edu.stanford.nlp.tagger.maxent.ExtractorMidSentenceCap),"
        "prefix(10),suffix(10),unicodeshapes(0),"
        "rareExtractor(edu.stanford.nlp.tagger.maxent.ExtractorNonAlphanumeric)"
    ),
    "closedClassTagThreshold": "40",
    "curWordMinFeatureThresh": "2",
    "rareWordMinFeatureThresh": "5",
    "rareWordThresh": "5",
    "search": "owlqn",
    "sigmaSquared": "0.5",
    "minFeatureThresh": "2",
    "tokenize": "true",
    "minWordsLockTags": "1"
}

PROPS_ARABIC = {
    "arch": "words(-2,2),order(1),prefix(6),suffix(6),unicodeshapes(1)",
    "learnClosedClassTags": "false",
    "minFeatureThresh": "3",
    "rareWordMinFeatureThresh": "3",
    "rareWordThresh": "5",
    "search": "owlqn",
    "sigmaSquared": "0.0"
}

PROPS_CHINESE = {
    "arch": (
        "generic,suffix(4),prefix(4),unicodeshapes(-1,1),"
        "unicodeshapeconjunction(-1,1),words(-2,-2),words(2,2),"
    ),
    "closedClassTagThreshold": "40",
    "curWordMinFeatureThresh": "1",
    "learnClosedClassTags": "false",
    "minFeatureThresh": "3",
    "rareWordMinFeatureThresh": "3",
    "rareWordThresh": "20",
    "sigmaSquared": "0.75",
    "tokenize": "false"
}

PROPS_SPANISH = {
    "arch": "left3words,naacl2003unknowns,allwordshapes(-1,1)",
    "closedClassTagThreshold": "40",
    "curWordMinFeatureThresh": "2",
    "learnClosedClassTags": "false",
    "minFeatureThresh": "2",
    "rareWordMinFeatureThresh": "10",
    "rareWordThresh": "5",
    "search": "owlqn2",
    "sigmaSquared": "0.75",
    "tokenize": "true",
    "tokenizerOptions": "asciiQuotes"
}

PROPS = {
    "en": PROPS_ENGLISH,
    "ar": PROPS_ARABIC,
    "zh": PROPS_CHINESE,
    "es": PROPS_SPANISH
}

SHARED_PROPS = {
    "veryCommonWordThresh": "250",
    "nthreads": "8",
    "regL1": "0.75",
    "sgml": "false",
    "tagSeparator" : "\\t",
    "encoding": "UTF-8",
    "trainFile": None,
    "lang": None,
    "iterations": None,
    "model": None
}

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
            train_set = data_archives.get_dataset_path(self.args.lang, self.args.treebank, "train")
            SHARED_PROPS["trainFile"] = f"format=TSV,wordColumn=0,tagColumn=1,{train_set}"
            SHARED_PROPS["lang"] = data_archives.LANGS_FULL[self.args.lang]
            SHARED_PROPS["iterations"] = self.args.iter
            SHARED_PROPS["model"] = self.model_path()
            props_lang = PROPS["lang"]
            props_lang.update(SHARED_PROPS)
            for key in props_lang:
                fp.write(f"{key} = {props_lang[key]}\n")

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

    def script_path_train(self):
        return "models/stanford/stanford-postagger.jar"

    def train_string(self):
        return (
            "java -cp [script_path_train] "
            "edu.stanford.nlp.tagger.maxent.MaxentTagger -props [model_base_path]/pocketML.props"
        )

    def predict_string(self):
        return ( 
            "java -mx2g -cp [script_path_test] "
            "edu.stanford.nlp.tagger.maxent.MaxentTagger -model [model_path] "
            "--encoding UTF-8 "
            "--testFile format=TSV,wordColumn=0,tagColumn=1,[dataset_test] [stdout] [pred_path]"
        )

    def code_size(self):
        base = "models/stanford/"
        code_files = [
            f"{base}/stanford-postagger.jar",
        ]
        total_size = int(JAVA_JRE_SIZE)
        for glob_str in code_files:
            files = glob(glob_str)
            for file in files:
                total_size += os.path.getsize(file)
        return total_size
