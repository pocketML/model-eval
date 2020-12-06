from glob import glob
from os import path
from util import data_archives

def find_value(lines, key):
    for line in lines:
        if key in line:
            split = line.split(":")
            return split[1].strip()
    return None

SORT_ORDER_MODEL = [
    "brill", "tnt", "crf", "hmm", "stanford", "svmtool",
    "bilstm_aux", "bilstm_crf", "flair", "meta_tagger", "bert_bpemb"
]

SORT_ORDER_LANG = [
    "am", "da", "en", "ar", "hi", "zh", "ru", "es", "tr", "vi"
]

def load_results():
    model_folders = glob("results/*")
    data = []
    for model_folder in model_folders:
        model_name = path.split(model_folder)[1]
        language_folders = glob(f"{model_folder}/*")
        language_data = []
        language_names = []
        for language_folder in language_folders:
            language_name = path.split(language_folder)[1].split("_")[0]

            filename = glob(f"{language_folder}/*")[-1]
            with open(filename, "r") as fp:
                lines = fp.readlines()
                token_acc = find_value(lines, "Final token acc")
                sentence_acc = find_value(lines, "Final sentence acc")
                memory_footprint = find_value(lines, "Memory usage")
                code_size = find_value(lines, "Code size")
                model_size = find_value(lines, "Model size")
                compressed_size = find_value(lines, "Compressed size")
                language_data.append((language_name, {
                    "token": token_acc, "sentence": sentence_acc,
                    "memory": memory_footprint, "code": code_size,
                    "model": model_size, "compressed": compressed_size
                }));
                language_names.append(language_name)

        missing_data = set(SORT_ORDER_LANG) - set(language_names)
        for lang in missing_data:
            language_data.append((lang, {
                "token": "?", "sentence": "?",
                "memory": "?", "code": "?",
                "model": "?", "compressed": "?"
            }))

        language_data.sort(key=lambda x: SORT_ORDER_LANG.index(x[0]))
        data.append((model_name, language_data))

    data.sort(key=lambda x: SORT_ORDER_MODEL.index(x[0]))
    return data

def format_size(size):
    unit = "KB"
    num = float(size)
    if num > 1000:
        num = num / 1000
        unit = "MB"
        if num > 1000:
            num = num / 1000
            unit = "GB"

    if num > 99:
        formatted = f"{num:.1f}"
    elif num > 9:
        formatted = f"{num:.2f} "
    else:
        formatted = f"{num:.3f}"
    if len(formatted) != 5:
        formatted = formatted + ("0" * (5 - len(formatted)))
    return formatted + " " + unit

def save_size_measurements():
    data = load_results()

    langs = []
    for lang in SORT_ORDER_LANG:
        lang_data = []
        for model, model_data in data:
            for lang_, data_dict in model_data:
                if lang_ == lang:
                    lang_data.append((model, data_dict))
        langs.append((lang, lang_data))

    model_sizes = []
    for key in ("memory", "code", "model", "compressed"):
        model_size = {m: [] for m in SORT_ORDER_MODEL}

        with open(f"formatted_results_{key}.txt", "w", encoding="utf-8") as fp:
            for lang, model_data in langs:
                for model, data_dict in model_data:
                    size = data_dict[key]
                    formatted = size
                    if size != "?":
                        model_size[model].append(float(size))
                        formatted = format_size(size)

                    line = f"{formatted}\t"
                    fp.write(line)
                fp.write("\n")
        
        model_sizes.append(model_size)

    reformatted = []
    for model_name in SORT_ORDER_MODEL:
        model_list = []
        for metric_data in model_sizes:
            for model_data in metric_data:
                if model_name == model_data:
                    avg = sum(metric_data[model_data]) / len(metric_data[model_data])
                    formatted = format_size(avg)
                    model_list.append(formatted)
        reformatted.append(model_list)

    with open(f"formatted_results_avg.txt", "w", encoding="utf-8") as fp:
        for model_data in reformatted:
            for metric_data in model_data:
                fp.write(f"{metric_data}\t")
            fp.write("\n")

def get_sentences(lang, treebank, dataset_type):
        data_path = data_archives.get_dataset_path(lang, treebank, dataset_type)
        data = open(data_path, "r", encoding="utf-8").readlines()
        sentences = []
        curr_sentences = []
        for line in data:
            if line.strip() == "":
                sentences.append(curr_sentences)
                curr_sentences = []
            else:
                curr_sentences.append(line.split(None)[0])
        return sentences

def flatten(arr):
    flattened = []
    for l in arr:
        flattened.extend(l)
    return flattened

def save_dataset_stats():
    for lang in SORT_ORDER_LANG:
        print(f"===== {lang} =====")
        treebank = data_archives.get_default_treebank(lang)
        embed_path = data_archives.get_embeddings_path(lang)

        train_sentence_list = get_sentences(lang, treebank, "train")
        train_sentences = len(train_sentence_list)
        train_token_list = flatten(train_sentence_list)
        train_tokens = len(train_token_list)

        test_sentence_list = get_sentences(lang, treebank, "test")
        test_sentences = len(test_sentence_list)
        test_token_list = flatten(test_sentence_list)
        test_tokens = len(test_token_list)

        dev_sentence_list = get_sentences(lang, treebank, "dev")
        dev_sentences = len(dev_sentence_list)
        dev_token_list = flatten(dev_sentence_list)
        dev_tokens = len(dev_token_list)

        total_sentences = train_sentences + test_sentences + dev_sentences
        total_tokens = train_tokens + test_tokens + dev_tokens

        test_not_in_train = len(set(test_token_list) - set(train_token_list))
        dev_not_in_train = len(set(dev_token_list) - set(train_token_list))

        test_not_in_train_ratio = test_not_in_train / train_tokens
        dev_not_in_train_ratio = dev_not_in_train / train_tokens

        print(f"Sentences : {total_sentences}")
        print(f"Tokens: {total_tokens}")
        print(f"Test words not in train: {(test_not_in_train_ratio * 100):.2f}%")
        print(f"Dev words not in train: {(dev_not_in_train_ratio * 100):.2f}%")

if __name__ == "__main__":
    save_dataset_stats()