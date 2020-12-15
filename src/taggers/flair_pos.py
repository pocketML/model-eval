from functools import lru_cache
import logging
import re
from flair.data import Sentence
from flair.models import SequenceTagger
from flair.embeddings import TokenEmbeddings, CharacterEmbeddings, StackedEmbeddings, WordEmbeddings
from flair.datasets import UniversalDependenciesCorpus, DataLoader
from flair.trainers import ModelTrainer
from flair import device
import torch
import numpy as np
from taggers.tagger_wrapper_import import ImportedTagger
from util import data_archives

class PolyglotEmbeddings(TokenEmbeddings):
    """
    Custom class for Polyglot Embeddings to conform to
    Flairs expected type of embeddings.
    """
    def __init__(self, lang):
        self.embeddings = "polyglot"
        self.name = self.embeddings
        self.static_embeddings = True
        self.field = None
        embeddings, dims = data_archives.load_embeddings(lang)
        self.precomputed_word_embeddings = embeddings
        self._embedding_length = dims

        super().__init__()

    @property
    def embedding_length(self):
        return self._embedding_length

    @lru_cache(maxsize=10000, typed=False)
    def get_cached_vec(self, word):
        if word in self.precomputed_word_embeddings:
            word_embedding = self.precomputed_word_embeddings[word]
        elif word.lower() in self.precomputed_word_embeddings:
            word_embedding = self.precomputed_word_embeddings[word.lower()]
        elif re.sub(r"\d", "#", word.lower()) in self.precomputed_word_embeddings:
            word_embedding = self.precomputed_word_embeddings[
                re.sub(r"\d", "#", word.lower())
            ]
        elif re.sub(r"\d", "0", word.lower()) in self.precomputed_word_embeddings:
            word_embedding = self.precomputed_word_embeddings[
                re.sub(r"\d", "0", word.lower())
            ]
        else:
            word_embedding = np.zeros(self.embedding_length, dtype="float")

        word_embedding = torch.tensor(
            word_embedding.tolist(), device=device, dtype=torch.float
        )
        return word_embedding

    def _add_embeddings_internal(self, sentences):

        for i, sentence in enumerate(sentences):

            for token, token_idx in zip(sentence.tokens, range(len(sentence.tokens))):

                if "field" not in self.__dict__ or self.field is None:
                    word = token.text
                else:
                    word = token.get_tag(self.field).value

                word_embedding = self.get_cached_vec(word=word)

                token.set_embedding(self.name, word_embedding)

        return sentences

    def __str__(self):
        return self.name

    def extra_repr(self):
        # fix serialized models
        if "embeddings" not in self.__dict__:
            self.embeddings = self.name

        return f"'{self.embeddings}'"

class ListenFilter(logging.Filter):
    def __init__(self, name, args):
        super().__init__(name=name)
        self.args = args
        self.curr_epoch = 0

    def filter(self, record):
        text = record.getMessage()
        if text.startswith("EPOCH"):
            split = text.split(" ")
            self.curr_epoch = int(split[1].strip())
        elif text.startswith("TEST"):
            split = text.split("score ")
            acc = float(split[1])

        if self.args.verbose:
            return True
        return False

class RequestsHandler(logging.Handler):
    def emit(self, record):
        pass

class Flair(ImportedTagger):
    def __init__(self, args, model_name, load_model=False):
        super().__init__(args, model_name, load_model)
        self.embeds_unsupported_langs = ("am", "vi")
        (data_folder, train_file, test_file, dev_file) = self.format_data("train")
        self.corpus = UniversalDependenciesCorpus(
            data_folder, train_file, test_file, dev_file, split_multiwords=False
        )
        dictionary = self.corpus.make_tag_dictionary("upos")
        if not load_model:
            embeddings = self.get_embeddings()

            self.model = SequenceTagger(hidden_size=256, embeddings=embeddings,
                                        tag_dictionary=dictionary, tag_type="upos",
                                        rnn_layers=2, use_crf=True)
        else:
            self.model.tag_dictionary = dictionary

    def get_embeddings(self):
        embeddings = [
            PolyglotEmbeddings(self.args.lang),
            CharacterEmbeddings()
        ]
        if not self.args.lang in self.embeds_unsupported_langs:
            embeddings.append(WordEmbeddings(self.args.lang))

        return StackedEmbeddings(embeddings=embeddings)

    def train(self, train_data):
        flair_logger = logging.getLogger("flair")
        handler = RequestsHandler()
        flair_logger.addHandler(handler)

        filter = ListenFilter("filter", self.args)
        flair_logger.addFilter(filter)

        trainer = ModelTrainer(self.model, self.corpus)

        trainer.train(self.model_base_path(),
                        learning_rate=0.1, mini_batch_size=32,
                        max_epochs=self.args.iter if self.args.max_iter else 100,
                        train_with_dev=True, monitor_test=True, embeddings_storage_mode="gpu")

    def format_data(self, dataset_type):
        if dataset_type == "train": # Flair expects URI paths to data when training.
            data_paths = data_archives.get_dataset_path(self.args.lang,
                                                        self.args.treebank,
                                                        None, simplified=False)
            path_only = "/".join(data_paths[0].split("/")[:-1])
            train_file = data_paths[0].split("/")[-1]
            test_file = data_paths[1].split("/")[-1]
            dev_file = data_paths[2].split("/")[-1]
            return (path_only, train_file, test_file, dev_file)
        else: # Use Flairs built-in DataLoader when loading test data.
            sentences = []
            for batch in DataLoader(self.corpus.test, 32, 8):
                for sentence in batch:
                    curr_sentence = []
                    for token in sentence:
                        true_tag = token.get_tag("upos").value
                        curr_sentence.append((token.text, true_tag))
                    sentences.append(curr_sentence)
            return sentences

    def predict(self, sentence):
        flair_sentence = Sentence()
        for word in sentence:
            flair_sentence.add_token(word)
        self.model.predict(flair_sentence, label_name="predicted")
        predictions = []
        for token in flair_sentence:
            predictions.append((token.get_tag("upos").value, token.get_tag("predicted").value))
        return predictions

    def code_size(self):
        torch_lib_size = 999081514 # 1.91 GB
        return super().code_size() + torch_lib_size

    def __getstate__(self):
        return (self.args, self.model_name)

    def __setstate__(self, state):
        args, model_name = state
        self.__init__(args, model_name, True)
