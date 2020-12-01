from taggers.tagger_wrapper_import import ImportedTagger
import nltk

class TnT(ImportedTagger):
    def __init__(self, args, model_name, load_model=False):
        super().__init__(args, model_name, load_model)
        if not load_model:
            self.model = nltk.TnT()
