from os import path
from glob import glob
import shutil

DATASETS = {
    "english": ("UD_English", "GUM")
}

def get_dataset_folder(language, treebank=None):
    language_name, default_treebank = DATASETS[language]
    if treebank is not None:
        default_treebank = treebank
    return f"data/treebanks/{language_name}-{default_treebank}"

def parse_dataset(dataset):
    tags_train_in = glob(f"{dataset}/*ud-train.conllu")[0]
    tags_test_in = glob(f"{dataset}/*ud-test.conllu")[0]
    tags_dev_in = glob(f"{dataset}/*ud-dev.conllu")[0]

    for tags in (tags_train_in, tags_test_in, tags_dev_in):
        file_name = tags.replace("\\", "/").split("/")[-1]
        new_file = f"{dataset}/simplified_{file_name}"
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
                        line_out = f"{word} {tag}"

                    file_out.write(line_out + "\n")

def get_data(language, treebank=None):
    treebanks_path = "data/treebanks"
    if not path.exists(treebanks_path):
        shutil.unpack_archive(f"{treebanks_path}.zip", treebanks_path)

    folder = get_dataset_folder(language, treebank)
    return parse_dataset(folder)

if __name__ == "__main__":
    get_data("english", "GUM")
