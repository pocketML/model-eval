from glob import glob
from os import unlink
from shutil import unpack_archive
import requests

LANGUAGES = {
    "en": "english",
    "english": "english",
    "da": "danish",
    "danish": "danish"
}

def download_and_unpack(archive):
    url = f"https://magnusjacobsen.com/projects/pocketml/{archive}.tgz"
    response = requests.get(url)

    filename = f"{archive}/{url.split('/')[-1]}"
    print(f"Downloading {archive} from {url}...")

    with open(filename, "wb") as fp:
        for chunk in response.iter_content(chunk_size=128):
            fp.write(chunk)

    try:
        unpack_archive(filename, f"{archive}/")
        unlink(filename) # Remove old archive.
    except ValueError:
        pass

def archive_exists(archive):
    return len(glob(archive + "/*")) > 1

def get_default_treebank(lang):
    lang_capitalized = LANGUAGES[lang].capitalize()
    folder_name = glob(f"data/UD_{lang_capitalized}-*")[0].replace("\\", "/").split("/")[-1]
    return folder_name.split("-")[-1].lower()

def get_dataset_path(lang, treebank, dataset_type):
    lang_capitalized = LANGUAGES[lang].capitalize()
    if treebank is None:
        treebank_upper = get_default_treebank(lang).upper()
    else:
        treebank_upper = treebank.upper()
    dataset_path = f"data/UD_{lang_capitalized}-{treebank_upper}/mini"
    path = glob(f"{dataset_path}/*-{dataset_type}.conllu")
    return path[0].replace("\\", "/")

if __name__ == "__main__":
    download_and_unpack("models")
