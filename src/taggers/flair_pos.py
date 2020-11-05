from functools import lru_cache
from six.moves import cPickle as pickle
import re
from flair.models import SequenceTagger
from flair.embeddings import WordEmbeddings, TokenEmbeddings
from flair.data import Dictionary
from flair.datasets import UniversalDependenciesCorpus
from flair.trainers import ModelTrainer
from flair import device
import torch
import numpy as np
from taggers.tagger_wrapper_import import ImportedTagger
import data_archives

class PolyglotEmbeddings(TokenEmbeddings):
    def __init__(self, embeddings_path):
        self.embeddings = "polyglot"
        self.name = self.embeddings
        self.static_embeddings = True
        self.field = None
        self.precomputed_word_embeddings = {}
        content = open(embeddings_path, "rb").read()
        data_tuple = pickle.loads(content, encoding="latin1")
        for word, vec in zip(data_tuple[0], data_tuple[1]):
            self.precomputed_word_embeddings[word] = vec
        print(len(self.precomputed_word_embeddings))
        self._embedding_length = len(data_tuple[1][0])

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

class FLAIR(ImportedTagger):
    def __init__(self, args, model_name, load_model=False):
        super().__init__(args, model_name, load_model)
        if not load_model:
            embedding_path = data_archives.get_embeddings_path(args.lang)
            print(embedding_path)
            embeddings = PolyglotEmbeddings(embedding_path)
            dictionary = Dictionary()
            dictionary.add_item("O")
            tags = data_archives.get_tags_in_dataset(args.lang, args.treebank, "train")
            for tag in tags:
                dictionary.add_item(tag)
            dictionary.add_item("<START>")
            dictionary.add_item("<STOP>")

            self.model = SequenceTagger(hidden_size=256, embeddings=embeddings,
                                        tag_dictionary=dictionary, tag_type="pos",
                                        use_crf=True)

    def train(self, train_data):
        print(train_data)
        (data_folder, train_file, test_file, dev_file) = train_data
        corpus = UniversalDependenciesCorpus(data_folder, train_file, test_file, dev_file)
        trainer = ModelTrainer(self.model, corpus)
        trainer.train(f"models/flair/{self.args.lang}_{self.args.treebank}",
                      learning_rate=0.1,
                      mini_batch_size=32,
                      max_epochs=self.args.iter)

    def format_data(self, dataset_type):
        data_paths = data_archives.get_dataset_path(self.args.lang,
                                                    self.args.treebank,
                                                    None, simplified=False)
        path_only = "/".join(data_paths[0].split("/")[:-1])
        train_file = data_paths[0].split("/")[-1]
        test_file = data_paths[1].split("/")[-1]
        dev_file = data_paths[2].split("/")[-1]
        return (path_only, train_file, test_file, dev_file)
