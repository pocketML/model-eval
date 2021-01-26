from math import sqrt
import os
from glob import glob

PROPER_MODEL_NAMES = {
    "bert_bpemb": "BERT-BPEmb",
    "bilstm_aux": "BiLSTM (Plank)",
    "bilstm_crf": "BiLSTM (Yasunaga)",
    "flair": "Flair",
    "svmtool": "SVMTool",
    "stanford": "Stanford Tagger",
    "tnt": "TnT",
    "hmm": "HMM",
    "brill": "Brill",
    "meta_tagger": "Meta-BiLSTM"
}

LANGS_FULL = {
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

LANGUAGE_ORDER = [
    "ar", "zh", "en", "es", "am", "da", "hi", "ru", "tr", "vi"
]
TAGGER_ORDER = [
    "brill", "tnt", "svmtool", "stanford", "hmm", "bilstm_aux",
    "bilstm_crf", "flair", "meta_tagger", "bert_bpemb"
]

def find_value(lines, key):
    for line in lines:
        if key in line:
            split = line.split(":")
            return split[1].strip()
    return None

def load_results():
    model_folders = list(filter(lambda x: "csv" not in x, glob("../results/*")))

    mapped_data = {}
    model_folders.sort(key=lambda x: TAGGER_ORDER.index(x.replace("\\", "/").split("/")[-1]))
    for model_folder in model_folders:
        if "csv" in model_folder:
            continue
        model_name = os.path.split(model_folder)[1]
        language_folders = glob(f"{model_folder}/*")
        language_folders.sort(key=lambda x: LANGUAGE_ORDER.index(os.path.split(x)[1].split("_")[0]))
        for language_folder in language_folders:
            language_name = os.path.split(language_folder)[1].split("_")[0]
            if language_name not in mapped_data:
                mapped_data[language_name] = {}

            filename = glob(f"{language_folder}/*")[-1]
            with open(filename, "r") as fp:
                lines = fp.readlines()
                token_acc = float(find_value(lines, "Final token acc")) * 100
                sentence_acc = float(find_value(lines, "Final sentence acc")) * 100
                memory_footprint = float(find_value(lines, "Memory usage"))
                code_size = float(find_value(lines, "Code size"))
                model_size = float(find_value(lines, "Model size"))
                compressed_size = float(find_value(lines, "Compressed size"))
                mapped_data[language_name][model_name] = {
                    "token": token_acc, "sentence": sentence_acc,
                    "memory": memory_footprint, "code": code_size,
                    "model": model_size, "compressed": compressed_size
                }

    return mapped_data

def create_csv_file(results, metric):

    with open(f"../results/csv/{metric}.csv", "w", encoding="utf-8") as fp:
        tagger_names = list(PROPER_MODEL_NAMES[tagger] for tagger in TAGGER_ORDER)
        header = "Language," + ",".join(tagger_names) + ",Avg."
        fp.write(header + "\n")
        metric_values = [[] for _ in TAGGER_ORDER]
        for language in results:
            lang_pretty = LANGS_FULL[language].capitalize() + ","
            values = []
            for index, tagger in enumerate(TAGGER_ORDER):
                model_data = results[language].get(tagger)
                value = 0
                if model_data is not None:
                    value = model_data[metric]
                values.append(value)
                metric_values[index].append(value)

            line = lang_pretty + ",".join(
                f"{val:.2f}" for val in values
            )
            line += f",{sum(values) / len(values):.2f}"
            fp.write(line + "\n")
        average_four = [f"{(sum(values[:4]) / 4):.2f}" for values in metric_values]
        average_total = [f"{sum(values) / len(values):.2f}" for values in metric_values]
        std_dev = []
        for values in metric_values:
            mean = sum(values) / len(values)
            std_dev_value = sqrt(sum((x - mean) ** 2 for x in values) / (len(values)-1))
            std_dev.append(f"{std_dev_value:.2f}")
        avg_four_line = "Avg. (4 lang.)," + ",".join(average_four)
        avg_all_line = "Average (all)," + ",".join(average_total)
        std_dev_line = "Std. dev. (all)," + ",".join(std_dev)
        fp.write(avg_four_line + "\n")
        fp.write(avg_all_line + "\n")
        fp.write(std_dev_line + "\n")

if __name__ == "__main__":
    results = load_results()
    if not os.path.exists("../results/csv"):
        os.mkdir("../results/csv")
    for metric in ("token", "sentence", "memory", "code", "model", "compressed"):
        create_csv_file(results, metric)
