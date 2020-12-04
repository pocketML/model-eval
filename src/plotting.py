from glob import glob
from os import path
from sys import argv
import argparse
import math
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from util.data_archives import LANGS_FULL, LANGS_ISO, get_default_treebank

SOUTH = 0
EAST = 1
NORTH = 2
WEST = 3
POS_OFFSETS = {
    "en_token_memory": [
        NORTH, WEST, WEST, EAST, EAST, NORTH,
        WEST, EAST, SOUTH, SOUTH, SOUTH
    ],
    "en_token_code": [
        EAST, NORTH, EAST, EAST, EAST, SOUTH,
        WEST, WEST, SOUTH, SOUTH, SOUTH
    ],
    "en_token_model": [
        NORTH, EAST, NORTH, EAST, WEST, SOUTH,
        WEST, NORTH, SOUTH, WEST, SOUTH
    ],
    "en_token_compressed": [
        NORTH, EAST, SOUTH, EAST, WEST, SOUTH,
        WEST, NORTH, SOUTH, WEST, SOUTH
    ]
}

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

def plot_data(data, axis, directions, acc_metric="token", size_metric="memory"):
    legend_text = {
        "token": "Token Accuracy", "sentence": "Sentence Accuracy",
        "memory": "Memory Footprint", "code": "Code Size",
        "model": "Uncompresse Model Size", "compressed": "Compressed Model Size"
    }
    axis.set_xlabel(f"{legend_text[size_metric]} (MB)")
    axis.set_ylabel(legend_text[acc_metric])

    sorted_data = []
    for model in data:
        model_data = data[model]
        accuracy = float(model_data[acc_metric])
        footprint = int(model_data[size_metric]) / 1000
        sorted_data.append((model, accuracy, footprint))

    sorted_data.sort(key=lambda x: x[2])

    for model, accuracy, footprint in sorted_data:
        axis.scatter(footprint, accuracy, label=model)

    for index, (model, accuracy, footprint) in enumerate(sorted_data):
        x, y = axis.transData.transform((footprint, accuracy))

        text_1 = f"{model}"
        text_2 = f"{footprint:.2f}MB"
        text_3 = f"{accuracy:.4f}"

        # Define offsets.
        text_max = max(len(text_1), len(text_2), len(text_3))
        offset_x = (12 + int(text_max))
        y_gap = 10
        offset_y_1 = 6
        offset_y_2 = offset_y_1 + y_gap
        offset_y_3 = offset_y_2 + y_gap

        # Center x and y on the data points (roughly).
        x = x - offset_x
        x += 8
        y += 2
        y_1 = y - offset_y_1
        y_2 = y - offset_y_2
        y_3 = y - offset_y_3

        if directions[index] == SOUTH: # Write text below data points.
            y_1 -= offset_y_1
            y_2 -= offset_y_1
            y_3 -= offset_y_1
        elif directions[index] == WEST: # Write text left of data points.
            x -= offset_x
            y_1 += y_gap
            y_2 += y_gap
            y_3 += y_gap
        elif directions[index] == NORTH:
            y_1 += offset_y_2 * 2
            y_2 += offset_y_2 * 2
            y_3 += offset_y_2 * 2
        elif directions[index] == EAST:
            x += offset_x
            y_1 += y_gap
            y_2 += y_gap
            y_3 += y_gap

        x_1, y_1 = axis.transData.inverted().transform((x, y_1))
        x_2, y_2 = axis.transData.inverted().transform((x, y_2))
        x_3, y_3 = axis.transData.inverted().transform((x, y_3))

        bbox = dict(boxstyle="square", fc="1", linewidth=0)

        axis.annotate(text_1, (x_1, y_1), xytext=(0, 0), xycoords="data", textcoords="offset points", fontweight=800)
        axis.annotate(text_2, (x_2, y_2), xytext=(0, 0), xycoords="data", textcoords="offset points", bbox=bbox)
        axis.annotate(text_3, (x_3, y_3), xytext=(0, 0), xycoords="data", textcoords="offset points", bbox=bbox)

    return sorted_data

def plot_pareto(data, axis):
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

    axis.grid(b=True, which="major", axis="both")
    axis.plot(points_x, points_y, linestyle=":", linewidth=2, c="red")

def plot_results(language, acc_metric, size_metric, save_to_file):
    results = load_results()
    treebank = get_default_treebank(language)
    fig, ax = plt.subplots()
    ax.set_title(f"{LANGS_FULL[language].capitalize()} ({treebank.upper()} Treebank)")
    ax.set_xscale("log")

    directions = POS_OFFSETS[f"{language}_{acc_metric}_{size_metric}"]
    sorted_data = plot_data(results[language], ax, directions, acc_metric, size_metric)
    plot_pareto(sorted_data, ax)

    manager = plt.get_current_fig_manager()
    w, h = manager.window.maxsize()
    if w > 1920:
        w = 1920
    manager.resize(w, h)
    fig.set_size_inches(17, 10)

    if save_to_file:
        filename = f"plots/{language}_{treebank}-{acc_metric}_{size_metric}.png"
        plt.savefig(filename)
    else:
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
    parser.add_argument(
        "-s", "--save", action="store_true", help="Save picture of plot to file"
    )

    args = parser.parse_args()

    plot_results(LANGS_ISO[args.language], args.accuracy_metric, args.size_metric, args.save)
