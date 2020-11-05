from taggers.tagger_wrapper_nltk import NLTKTagger
import data_archives
from flare.models import SequenceTagger
from flare.embeddings import WordEmbeddings
from flare.data import Dictionary

class FLARE(NLTKTagger):
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
