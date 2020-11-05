from taggers.tagger_wrapper_nltk import NLTKTagger
import nltk

class HMM(NLTKTagger):
    def __init__(self, args, model_name, load_model=False):
        super().__init__(args, model_name, load_model)
        if not load_model:
            self.model = None

    def train(self, train_data):
        return nltk.HiddenMarkovModelTagger.train(train_data) # NLTK is weird...

    def __getstate__(self):
        return (self.args, self.model_name)

    def __setstate__(self, state):
        args, model_name = state
        self.__init__(args, model_name, True)
