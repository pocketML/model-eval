from glob import glob
from os import path
from sys import argv
import argparse
import math
import matplotlib.pyplot as plt
from util.data_archives import LANGS_FULL, LANGS_ISO, get_default_treebank

def find_value(lines, key):
    for line in lines:
        if key in line:
            split = line.split(":")
            return split[1].strip()
    return None

def load_results():
    model_folders = glob("results/*")
    mapped_data = {}
    for model_folder in model_folders:
        model_name = path.split(model_folder)[1]
        language_folders = glob(f"{model_folder}/*")
        for language_folder in language_folders:
            language_name = path.split(language_folder)[1].split("_")[0]
            if language_name not in mapped_data:
                mapped_data[language_name] = {}

            filename = glob(f"{language_folder}/*")[-1]
            with open(filename, "r") as fp:
                lines = fp.readlines()
                token_acc = find_value(lines, "Final token acc")
                sentence_acc = find_value(lines, "Final sentence acc")
                memory_footprint = find_value(lines, "Memory usage")
                code_size = find_value(lines, "Code size")
                model_size = find_value(lines, "Model size")
                compressed_size = find_value(lines, "Compressed size")
                mapped_data[language_name][model_name] = {
                    "token": token_acc, "sentence": sentence_acc,
                    "memory": memory_footprint, "code": code_size,
                    "model": model_size, "compressed": compressed_size
                }

    return mapped_data

def plot_data(data, acc_metric="token", size_metric="memory"):
    legend_text = {
        "token": "Token Accuracy", "sentence": "Sentence Accuracy",
        "memory": "Memory Footprint", "code": "Code Size",
        "model": "Uncompresse Model Size", "compressed": "Compressed Model Size"
    }
    plt.xlabel(f"{legend_text[size_metric]} (MB)")
    plt.ylabel(legend_text[acc_metric])

    sorted_data = []
    for model in data:
        model_data = data[model]
        accuracy = float(model_data[acc_metric])
        footprint = int(model_data[size_metric]) / 1000
        sorted_data.append((model, accuracy, footprint))

    sorted_data.sort(key=lambda x: x[2])

    for model, accuracy, footprint in sorted_data:
        x, y = footprint, accuracy
        offset_x = 500
        offset_y = 0.004
        if model == "brill":
            y += 0.008
        plt.text(x + offset_x, y - offset_y, f"Size = {x} MB")
        plt.text(x + offset_x, y - offset_y * 2, f"Acc = {y}")
        plt.scatter(footprint, accuracy, label=model)

    plt.legend()
    return sorted_data

def plot_pareto(data):
    points_x = []
    min_y = min(x[1] for x in data)
    min_y = (math.floor(min_y * 100.0)) / 100.0
    points_y = [min_y]
    highest_acc = 0
    for _, accuracy, footprint in data:
        y = highest_acc
        if accuracy > highest_acc:
            highest_acc = accuracy
            y = accuracy
        points_x.extend([footprint] * 2)
        points_y.extend([y] * 2)

    points_x.append(points_x[-1])

    plt.grid(b=True, which="major", axis="both")
    plt.plot(points_x, points_y, linestyle=":", linewidth=2)

def plot_results(language, acc_metric, size_metric):
    results = load_results()
    treebank = get_default_treebank(language)
    plt.title(f"{LANGS_FULL[language].capitalize()} ({treebank.upper()} Treebank)")
    sorted_data = plot_data(results[language], acc_metric, size_metric)
    plot_pareto(sorted_data)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot skyline/pareto curve of accuracy vs. size of POS taggers")

    choices_langs = LANGS_FULL.keys()
    parser.add_argument("language", type=str, choices=choices_langs, help="language to plot data for")
    parser.add_argument(
        "-am", "--accuracy-metric",
        choices=["token", "sentence"], default="token",
        help="The accuracy metric of the taggers"
    )
    parser.add_argument(
        "-sm", "--size-metric",
        choices=["memory", "code", "model", "compressed"],
        default="memory", help="The size metric of the taggers"
    )

    args = parser.parse_args()

    plot_results(LANGS_ISO[args.language], args.accuracy_metric, args.size_metric)
