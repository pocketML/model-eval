from glob import glob
from six.moves import cPickle as pickle
from os import unlink, path, mkdir, rename, walk
from shutil import unpack_archive
import requests

LANGS_FULL = { # Map from language ISO code or name -> Language name
    iso: lang for iso, lang in
    map(
        lambda line: tuple(line.strip().split(",")),
        open("data/langs_names.csv").readlines()
    )
}

LANGS_ISO = { # Map from language ISO code or name -> Language ISO code
    lang: iso for lang, iso in
    map(
        lambda line: tuple(line.strip().split(",")),
        open("data/langs_iso.csv").readlines()
    )
}

def download_and_unpack(archive_type, archive):
    """
    Download a zip archive, containing either a tagger or a dataset.
    Saves the arcive to:
    - models/[model_name] for models.
    - data/[language_name] for datasets.
    """
    url = f"https://magnusjacobsen.com/projects/pocketml/{archive_type}/{archive}.tgz"
    response = requests.get(url)

    archive_name = url.split("/")[-1]
    name = archive_name.split(".")[0]

    archive_path = f"{archive_type}/{archive_name}"
    folder_path = f"{archive_type}/{name}"

    print(f"Downloading {archive_type} archive '{archive}'...")
    print(f"Saving archive to {archive_path}...")

    with open(archive_path, "wb") as fp:
        for chunk in response.iter_content(chunk_size=128):
            fp.write(chunk)

    try:
        unpack_archive(archive_path, f"{archive_type}/")
        if archive_type == "data":
            dataset_file = glob(f"{folder_path}/UD_*")[0]
            rename(dataset_file, dataset_file.lower())
        unlink(archive_path) # Remove old archive.
    except ValueError:
        pass

def archive_exists(archive_type, archive):
    return len(glob(f"{archive_type}/{archive}/*")) > 1

def get_default_treebank(lang):
    language = LANGS_FULL[lang]
    folder_name = glob(f"data/{language}/ud_{language}-*")[0].replace("\\", "/").split("/")[-1]
    return folder_name.split("-")[-1].lower()

def get_dataset_folder_path(lang, treebank, simplified=True, eos=False):
    language = LANGS_FULL[lang]
    if treebank is None:
        treebank = get_default_treebank(lang)
    dataset_path = f"data/{language}/ud_{language}-{treebank}"
    if simplified:
        dataset_path += "/simplified"
    elif eos:
        dataset_path += "/simplified_eos"
    return dataset_path

def get_dataset_path(lang, treebank, dataset_type=None, simplified=True, eos=False):
    """
    Find and return the full path to a dataset for the given language and treebank.
    'dataset_type' can be train, test, or dev.
    'simplfiied' indicates whether to return original UD dataset, or simplified set with
    only (word, tag) pairs.
    'eos' indicates whether to return simplified dataset, where sentences are guaranteed
    to be terminated by a punctuation character.
    """
    dataset_path = get_dataset_folder_path(lang, treebank, simplified=simplified, eos=eos)
    glob_str = f"-{dataset_type}" if dataset_type is not None else ""
    paths = glob(f"{dataset_path}/*{glob_str}.conllu")

    if paths == []:
        dataset_name = "" if dataset_type is None else f"'{dataset_type.capitalize()}'"
        raise FileNotFoundError(
            f"{dataset_name} dataset for '{LANGS_FULL[lang].capitalize()}' " +
            f"using '{treebank}' treebank was not found."
        )

    for index, path_str in enumerate(paths):
        paths[index] = path_str.replace("\\", "/")

    if len(paths) > 1:
        sort_order = {"train.conllu": 0, "test.conllu": 1, "dev.conllu": 2}
        paths.sort(key=lambda x: sort_order[x.split("-")[-1]])

    return paths[0] if len(paths) == 1 else paths

def tagset_mapping(lang, treebank, dataset_type, from_complex=True):
    """
    Return a dictionary mapping XPOS tags to UPOS tags,
    for a dataset with the given type and language,
    or the other way around if 'from_complex' is false.
    """
    tag_mapping = {}
    dataset_full = get_dataset_path(lang, treebank, dataset_type, False)
    with open(dataset_full, "r", encoding="utf-8") as fp:
        for line in fp.readlines():
            stripped = line.strip()
            if stripped == "" or stripped.startswith("#"):
                continue
            split = stripped.split("\t")
            simple_tag = split[3]
            full_tag = split[4]
            if full_tag == "_":
                full_tag = simple_tag
            if from_complex:
                tag_mapping[full_tag] = simple_tag
            else:
                tag_mapping[simple_tag] = full_tag
    return tag_mapping

def get_embeddings_path(lang):
    """
    Get path of polyglot embeddings for a given language.
    """
    language = LANGS_FULL[lang]
    return f"data/{language}/embeddings/polyglot-{lang}.pkl"

def load_embeddings(lang):
    """
    Read polyglot embeddings and return embeddings as a dictionary.
    """
    content = open(get_embeddings_path(lang), "rb").read()
    words, vecs = pickle.loads(content, encoding="latin1")
    emb_dict = {}
    for word, vec in zip(words, vecs):
        emb_dict[word] = vec
    return emb_dict, len(vecs[0])

def transform_data(dataset):
    """
    Transform a dataset from full XPOS tagset to a simplified set
    containing just word and tag pairs.
    """
    tags_train_in = glob(f"{dataset}/*ud-train.conllu")[0]
    tags_test_in = glob(f"{dataset}/*ud-test.conllu")[0]
    tags_dev_in = glob(f"{dataset}/*ud-dev.conllu")[0]

    new_dir = f"{dataset}/simplified"
    if not path.exists(new_dir):
        mkdir(new_dir)

    for tags in (tags_train_in, tags_test_in, tags_dev_in):
        file_name = tags.replace("\\", "/").split("/")[-1]
        new_file = f"{new_dir}/{file_name}"
        with open(new_file, "w", encoding="utf-8") as file_out:
            with open(tags, "r", encoding="utf-8") as file_in:
                for line in file_in.readlines():
                    if line.startswith("#"):
                        continue

                    stripped = line.strip()

                    if stripped == "": # New sentence
                        line_out = ""
                    else:
                        split = stripped.split("\t")
                        word = split[1]
                        tag = split[3]
                        if tag == "_":
                            continue
                        line_out = f"{word}\t{tag}"

                    file_out.write(line_out + "\n")

# SVMTool does not like tabs as separators, or sentences that don't end with either . ! or ?
# so we:
# -- replace " " with "_" for the multiword tokens (in Vietnamese)
# -- replace "\t" with " "
# -- adds ". PUNCT" to the end of each sentence, if it doesn't have . ! or ?
def create_simplified_eos(dataset):
    tags_train_in = glob(f"{dataset}/simplified/*ud-train.conllu")[0]
    tags_test_in = glob(f"{dataset}/simplified/*ud-test.conllu")[0]
    tags_dev_in = glob(f"{dataset}/simplified/*ud-dev.conllu")[0]

    new_dir = f"{dataset}/simplified_eos"
    if not path.exists(new_dir):
        mkdir(new_dir)

    for tags in (tags_train_in, tags_test_in, tags_dev_in):
        file_name = tags.replace("\\", "/").split("/")[-1]
        new_file = f"{new_dir}/{file_name}"
        with open(new_file, "w", encoding="utf-8") as file_out:
            with open(tags, "r", encoding="utf-8") as file_in:
                prev_token = ""
                for line in file_in.readlines():
                    line_out = line.strip()         

                    if " " in line_out:
                        line_out = line_out.replace(" ", "_")

                    line_out = line_out.replace("\t", " ")

                    # Check if need an artificial sentence-end
                    if line_out == "" and (prev_token not in ".!?"):
                        line_out = ". PUNCT\n"

                    if line_out != "":
                        prev_token = line_out.split(" ")[0]
                    else:
                        prev_token = line_out

                    file_out.write(line_out + "\n")

# UD2 has this really nice feature, where there are non-tag multitoken words in the text
# most of our taggers do not like this
# so we just remove them from the conllu files
# fight me!
def remove_multitoken(folder):
    paths = glob(f'{folder}/*.conllu')
    for f in paths:
        newlines = []
        with open(f, "r", encoding="utf-8") as content:
            lines = content.readlines()
            for line in lines:
                if len(line.split(None)) > 1 and '-' in line.split(None)[0]:
                    continue
                newlines.append(line)
        with open(f, "w", encoding="utf-8") as content:
            for newline in newlines:
                content.write(newline)

def transform_dataset(language):
    treebanks = glob(f"data/{language}/ud_*")
    for folder in treebanks:
        print(f"Transforming {folder}")
        # removing all multitoken lines, of the form 1-2	Tirarla	_	_	_	_	_	_	_
        remove_multitoken(folder)
        transform_data(folder)
        create_simplified_eos(folder)

def get_embeddings_size(lang):
    return path.getsize(get_embeddings_path(lang))

def get_folder_size(folder_path):
    total_size = 0
    for dirpath, _, filenames in walk(folder_path):
        for f in filenames:
            fp = path.join(dirpath, f)
            total_size += path.getsize(fp)
    return total_size

def validate_simplified_datasets():
    possible_tags = {
        "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
        "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X", "_"
    }

    for lang in set(LANGS_ISO.values()):
        lang_full = LANGS_FULL[lang]
        print(f"====== {lang_full} =====")
        if not archive_exists("data", lang_full):
            download_and_unpack("data", lang_full)
            transform_dataset(lang_full)

        treebank = get_default_treebank(lang)
        for dataset_type in ("train", "test", "dev"):
            dataset_path = get_dataset_path(lang, treebank, dataset_type, True)
            with open(dataset_path, "r", encoding="utf-8") as fp:
                for index, line in enumerate(fp, start=1):
                    stripped = line.strip()
                    if stripped == "":
                        continue
                    split = stripped.split("\t")
                    if split[1] not in possible_tags:
                        print(f"Illegal tag '{split[1]}' on line {index} in {dataset_type} set!!!!")
