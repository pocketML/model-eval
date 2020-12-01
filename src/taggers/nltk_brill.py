from taggers.tagger_wrapper_import import ImportedTagger
import nltk
from nltk.tbl.template import Template
from nltk.tag.brill import Word, Pos
from taggers.nltk_hmm import HMM

class Brill(ImportedTagger):
    def __init__(self, args, model_name, load_model=False):
        super().__init__(args, model_name, load_model)
        if not load_model:
            base_tagger = HMM(args, "hmm", load_model=True)
            if not base_tagger.saved_model_exists():
                raise FileNotFoundError(f"Brill base tagger '{base_tagger.model_name}' missing!")

            features = [
                Template(Pos([-1])),
                Template(Pos([1])),
                Template(Pos([-2])),
                Template(Pos([2])),
                Template(Pos([-2, -1])),
                Template(Pos([1, 2])),
                Template(Pos([-3, -2, -1])),
                Template(Pos([1, 2, 3])),
                Template(Pos([-1]), Pos([1])),
                Template(Word([-1])),
                Template(Word([1])),
                Template(Word([-2])),
                Template(Word([2])),
                Template(Word([-2, -1])),
                Template(Word([1, 2])),
                Template(Word([-3, -2, -1])),
                Template(Word([1, 2, 3])),
                Template(Word([-1]), Word([1])),
                ]
            self.model = nltk.BrillTaggerTrainer(base_tagger.model, features)

    def __getstate__(self):
        return (self.args, self.model_name)

    def __setstate__(self, state):
        args, model_name = state
        self.__init__(args, model_name, True)
