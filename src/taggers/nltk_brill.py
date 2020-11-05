from taggers.tagger_wrapper_nltk import NLTKTagger
import nltk
from nltk.tbl.template import Template
from nltk.tag.brill import Word, Pos
from taggers.nltk_hmm import HMM

class Brill(NLTKTagger):
    def __init__(self, args, model_name, load_model=False):
        super().__init__(args, model_name, load_model)
        if not load_model:
            base_tagger = HMM(args, "hmm", load_model=True)
            features = [
                Template(Pos([-1])), Template(Pos([1])), Template(Word([0]))
            ]
            self.nltk_model = nltk.BrillTaggerTrainer(base_tagger.nltk_model, features)

    def __getstate__(self):
        return (self.args, self.model_name)

    def __setstate__(self, state):
        args, model_name = state
        self.__init__(args, model_name, True)
