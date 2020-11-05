from taggers.tagger_wrapper_import import ImportedTagger
import data_archives
from flair.models import SequenceTagger
from flair.embeddings import WordEmbeddings
from flair.data import Dictionary
from flair.datasets import UniversalDependenciesCorpus
from flair.trainers import ModelTrainer

class FLAIR(ImportedTagger):
    def __init__(self, args, model_name, load_model=False):
        super().__init__(args, model_name, load_model)
        if not load_model:
            embedding_path = data_archives.get_embeddings_path(args.lang)
            embeddings = WordEmbeddings(embeddings=embedding_path)
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
        path_only = data_paths[0].split("/")[:-1]
        train_file = data_paths[0].split("/")[-1]
        test_file = data_paths[1].split("/")[-1]
        dev_file = data_paths[2].split("/")[-1]
        return (path_only, train_file, test_file, dev_file)
