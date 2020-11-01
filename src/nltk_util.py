from nltk.tag import untag
import os
import dill as pickle
import data_archives

MODEL_PATH = "models/nltk_pickled"

def saved_model_exists(model_name):
    return os.path.exists(f"{MODEL_PATH}/{model_name}.pk")

def save_model(model, model_name):
    with open(f"{MODEL_PATH}/{model_name}.pk", "wb") as fp:
        pickle.dump(model, fp)

def load_model(model_name):
    with open(f"{MODEL_PATH}/{model_name}.pk", "rb") as fp:
        return pickle.load(fp)

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

def evaluate(model, test_data, pipe):
    correct = 0
    total = 0
    prev_pct = 0
    for index, sentence in enumerate(test_data):
        only_words = untag(sentence)
        preds = model.tag(only_words)
        for test_tup, pred_tup in zip(sentence, preds):
            if test_tup[1] == pred_tup[1]:
                correct += 1
            total += 1
        pct = int((index / len(test_data)) * 100)
        if pct > prev_pct:
            prev_pct = pct
            print(f"{pct}%", end="\r", flush=True)
    pipe.send(correct / total)
