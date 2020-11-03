from taggers.tagger_wrapper_nltk import NLTKTagger
import nltk

class TnT(NLTKTagger):
    def __init__(self, args, model_name, load_model=False):
        super().__init__(model_name, args, load_model)
        if not load_model:
            self.nltk_model = nltk.TnT()