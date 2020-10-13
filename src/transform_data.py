from os import path, mkdir
from glob import glob
import shutil

def parse_and_transform(dataset):
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
                        line_out = f"{word}\t{tag}"

                    file_out.write(line_out + "\n")

def transform_datasets():
    treebanks = glob("data/*")
    for folder in treebanks:
        parse_and_transform(treebanks)

if __name__ == "__main__":
    transform_datasets()
