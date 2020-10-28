import nltk
import data_archives

def format_nltk_data(args, dataset_type):
    data_path = data_archives.get_dataset_path(args.lang, args.treebank, dataset_type)
    train_data = open(data_path, "r", encoding="utf-8").readlines()
    sentences = []
    curr_senteces = []
    for line in train_data:
        if line.strip() == "":
            sentences.append(curr_senteces)
            curr_senteces = []
        else:
            curr_senteces.append(line.split(None))
    return sentences

async def evaluate(model, test_data):
    correct = 0
    total = 0
    for sentence in test_data:
        preds = model.tag_sents(sentence)
        for pred_tup in preds:
            if pred_tup[0][1] == pred_tup[1][0]:
                correct += 1
            total += 1
    return correct / total
