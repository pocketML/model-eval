from taggers.tagger_wrapper_import import ImportedTagger
import nltk

class CRF(ImportedTagger):
    def __init__(self, args, model_name, load_model=False):
        features = self.word_features
        train_opts = {
            "c1": 1.0,
            "c2": 1e-3,
            "max_iterations": args.iter,
            "feature.possible_transitions": True,
            "num_memories": 10
        }
        self.model = nltk.CRFTagger(features, False, train_opts)
        super().__init__(args, model_name, load_model)

    def train(self, train_data):
        return self.model.train(train_data, self.model_path())

    def save_model(self):
        pass # Model is saved during training by CRFSuite.

    def model_path(self):
        return f"{self.model_base_path()}/model.crfsuite"

    def load_model(self):
        self.model.set_model_file(self.model_path())

    def word_features(self, sentence, i):
        word = sentence[i]
        features = [
            'bias',
            'word.lower=' + word.lower(),
            'word[-3:]=' + word[-3:],
            'word[-2:]=' + word[-2:],
            'word.isupper=%s' % word.isupper(),
            'word.istitle=%s' % word.istitle(),
            'word.isdigit=%s' % word.isdigit()
        ]
        if i > 0:
            word1 = sentence[i-1]
            features.extend([
                '-1:word.lower=' + word1.lower(),
                '-1:word.istitle=%s' % word1.istitle(),
                '-1:word.isupper=%s' % word1.isupper()
            ])
        else:
            features.append('BOS')

        if i < len(sentence)-1:
            word1 = sentence[i+1]
            features.extend([
                '+1:word.lower=' + word1.lower(),
                '+1:word.istitle=%s' % word1.istitle(),
                '+1:word.isupper=%s' % word1.isupper()
            ])
        else:
            features.append('EOS')

        return features

    def __getstate__(self):
        return (self.args, self.model_name)

    def __setstate__(self, state):
        args, model_name = state
        self.__init__(args, model_name, True)
