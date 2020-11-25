from glob import glob
from os import unlink, path, mkdir, rename, walk
from shutil import unpack_archive
import requests

LANGS_FULL = { # Map from language ISO code or name -> Language name
    "am": "amharic",
    "amharic": "amharic",
    "da": "danish",
    "danish": "danish",
    "en": "english",
    "english": "english",
    "ar": "arabic",
    "arabic": "arabic",
    "hi": "hindi",
    "hindi": "hindi",
    "zh": "chinese",
    "chinese": "chinese",
    "ru": "russian",
    "russian": "russian",
    "es": "spanish",
    "spanish": "spanish",
    "tr": "turkish",
    "turkish": "turkish",
    "vi": "vietnamese",
    "vietnamese": "vietnamese"
}

LANGS_ISO = { # Map from language ISO code or name -> Language ISO code
    "am": "am",
    "amharic": "am",
    "da": "da",
    "danish": "da",
    "en": "en",
    "english": "en",
    "ar": "ar",
    "arabic": "ar",
    "hi": "hi",
    "hindi": "hi",
    "zh": "zh",
    "chinese": "zh",
    "ru": "ru",
    "russian": "ru",
    "es": "es",
    "spanish": "es",
    "tr": "tr",
    "turkish": "tr",
    "vi": "vi",
    "vietnamese": "vi"
}

def download_and_unpack(archive_type, archive):
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

def get_dataset_folder_path(lang, treebank, simplified=True):
    language = LANGS_FULL[lang]
    if treebank is None:
        treebank = get_default_treebank(lang)
    dataset_path = f"data/{language}/ud_{language}-{treebank}"
    if simplified:
        dataset_path += "/simplified"
    return dataset_path

def get_dataset_path(lang, treebank, dataset_type=None, simplified=True):
    dataset_path = get_dataset_folder_path(lang, treebank, simplified=simplified)
    glob_str = f"-{dataset_type}" if dataset_type is not None else ""
    paths = glob(f"{dataset_path}/*{glob_str}.conllu")

    for index, path_str in enumerate(paths):
        paths[index] = path_str.replace("\\", "/")

    if len(paths) > 1:
        sort_order = {"train.conllu": 0, "test.conllu": 1, "dev.conllu": 2}
        paths.sort(key=lambda x: sort_order[x.split("-")[-1]])

    return paths[0] if len(paths) == 1 else paths

def tagset_mapping(lang, treebank, dataset_type, from_complex=True):
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
            if from_complex:
                tag_mapping[full_tag] = simple_tag
            else:
                tag_mapping[simple_tag] = full_tag
    return tag_mapping

def get_embeddings_path(lang):
    language = LANGS_FULL[lang]
    return f"data/{language}/embeddings/polyglot-{lang}.pkl"

def transform_data(dataset):
    tags_train_in = glob(f"{dataset}/*ud-train.conllu")[0]
    tags_test_in = glob(f"{dataset}/*ud-test.conllu")[0]
    tags_dev_in = glob(f"{dataset}/*ud-dev.conllu")[0]

    new_dir = f"{dataset}/simplified"
    if not path.exists(new_dir):
        mkdir(new_dir)

    for tags in (tags_train_in, tags_test_in, tags_dev_in):
        file_name = tags.replace("\\", "/").split("/")[-1]
        new_file = f"{new_dir}/{file_name}"
        with open(new_file, "w", encoding="UTF-8") as file_out:
            with open(tags, "r", encoding="UTF-8") as file_in:
                for line in file_in.readlines():
                    if line.startswith("#"):
                        continue
                    split = line.split(None)

                    if split == []: # New sentence
                        line_out = ""
                    else:
                        word = split[1]
                        tag = split[3]
                        if tag == "_":
                            continue
                        line_out = f"{word}\t{tag}"

                    file_out.write(line_out + "\n")

def transform_dataset(language):
    treebanks = glob(f"data/{language}/UD_*")
    for folder in treebanks:
        print(f"Transforming {folder}")
        transform_data(folder)

def get_embeddings_size(lang):
    return path.getsize(get_embeddings_path(lang))

def get_folder_size(folder_path):
    total_size = 0
    for dirpath, _, filenames in walk(folder_path):
        for f in filenames:
            fp = path.join(dirpath, f)
            total_size += path.getsize(fp)
    return total_size
