import data_archives

def format_nltk_data(language, treebank, dataset_type):
    data_path = data_archives.get_dataset_path(language, treebank, dataset_type)
    train_data = open(data_path, "r", encoding="utf-8").readlines()
    sentences = []
    curr_senteces = []
    for line in train_data:
        if line.strip() == "":
            sentences.append(curr_senteces)
            curr_senteces = []
        else:
            curr_senteces.append(tuple(line.split(None)))
    return sentences
