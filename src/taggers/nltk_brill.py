from taggers.tagger_wrapper_nltk import NLTKTagger
import nltk
from nltk.tbl.template import Template
from nltk.tag.brill import Word, Pos
from taggers.nltk_tnt import TnT

class Brill(NLTKTagger):
    def __init__(self, args, model_name, load_model=False):
        super().__init__(args, model_name, load_model)
        if not load_model:
            base_tagger = TnT("tnt", args, load_model=True)
            features = [
                Template(Pos([-1])), Template(Pos([1])), Template(Pos([-2])),
                Template(Pos([2])), Template(Word([0])), Template(Word([1, -1]))
            ]
            self.nltk_model = nltk.BrillTaggerTrainer(base_tagger.nltk_model, features)