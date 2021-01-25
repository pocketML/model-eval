from glob import glob
import os
import argparse
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker 
#font = {'size': 24}
#plt.rc('font', **font)
from adjustText import adjust_text
from util.data_archives import LANGS_FULL, LANGS_ISO

SOUTH = 0
EAST = 1
NORTH = 2
WEST = 3
ARROW_SOUTH = 4
ANNOTATE_VALUES = False
SHARED_PLOT = False
INCLUDE_STANFORD = False

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
        model_name = os.path.split(model_folder)[1]
        language_folders = glob(f"{model_folder}/*")
        for language_folder in language_folders:
            language_name = os.path.split(language_folder)[1].split("_")[0]
            if language_name not in mapped_data:
                mapped_data[language_name] = {}

            filename = glob(f"{language_folder}/*")[-1]
            with open(filename, "r") as fp:
                lines = fp.readlines()
                token_acc = float(find_value(lines, "Final token acc"))
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

def get_data_for_language(data, lang, acc_metric, size_metric):
    if lang != "avg":
        return data[lang]

    languages = set(LANGS_ISO.values())

    averaged_acc = {m: [] for m in (data["en"] if INCLUDE_STANFORD else data["vi"])}
    averaged_size = {m: [] for m in (data["en"] if INCLUDE_STANFORD else data["vi"])}

    for lang in languages:
        if INCLUDE_STANFORD and not "stanford" in data[lang].keys():
            continue # Only include the 4 languages where Stanford Tagger can be used.

        for model_name in data[lang]:
            if INCLUDE_STANFORD or model_name != "stanford":
                averaged_acc[model_name].append(float(data[lang][model_name][acc_metric]))
                averaged_size[model_name].append(float(data[lang][model_name][size_metric]))

    for model_name in averaged_acc:
        averaged_acc[model_name] = sum(averaged_acc[model_name]) / len(averaged_acc[model_name])
        averaged_size[model_name] = sum(averaged_size[model_name]) / len(averaged_size[model_name])

    return {
        model: {
            acc_metric: averaged_acc[model],
            size_metric: averaged_size[model]
        } for model in averaged_acc
    }

def sort_data_by_size(data, acc_metric, size_metric):
    sorted_data = []
    for model in data:
        model_data = data[model]
        accuracy = float(f"{model_data[acc_metric]:.2f}")
        footprint = int(model_data[size_metric])
        sorted_data.append((model, accuracy, footprint))

    return sorted(sorted_data, key=lambda x: x[2])

def add_margins(axis, margin, acc_metric):
    x_limits = axis.get_xlim()
    y_limits = axis.get_ylim()

    x_min_pixel, y_min_pixel = axis.transData.transform((x_limits[0], y_limits[0]))
    x_max_pixel, y_max_pixel = axis.transData.transform((x_limits[1], y_limits[1]))

    x_min_pixel -= margin
    digits = math.log(x_limits[0], 10)
    x_limit = 10 ** (digits - 1)
    y_min_pixel -= margin
    x_max_pixel += margin
    y_max_pixel += margin

    _, min_y = axis.transData.inverted().transform((x_min_pixel, y_min_pixel))
    max_x, _ = axis.transData.inverted().transform((x_max_pixel, y_max_pixel))

    axis.set_xlim(x_limit, max_x)
    max_y = 1 if acc_metric == "token" else 0.6
    axis.set_ylim(min_y, max_y)

def plot_data(
        sorted_data, models_on_skyline, axis, plot_id,
        acc_metric="token", size_metric="memory"
):
    legend_text = {
        "token": "Token Accuracy", "sentence": "Sentence Accuracy",
        "memory": "Memory Footprint", "code": "Code Size",
        "model": "Size of Uncompressed Model Files", "compressed": "Size of Compressed Model Files"
    }
    axis.set_xlabel(f"{legend_text[size_metric]} (KB)", fontsize="medium")
    axis.set_ylabel(legend_text[acc_metric], fontsize="medium")

    colors = {
        "tnt": "#4b6238",
        "brill": "#7349c0",
        "crf": "#91d352",
        "hmm": "#c74a9f",
        "svmtool": "#8dd0a6",
        "stanford": "#c4483d",
        "bilstm_crf": "#6ba3bf",
        "bilstm_aux": "#cba748",
        "flair": "#4b2e4a",
        "meta_tagger": "#ae7b66",
        "bert_bpemb": "#b593c7"
    }

    models = [model for model, _, _ in sorted_data]
    accuracies = [accuracy for _,accuracy,_ in sorted_data]
    footprints = [footprint for _, _, footprint in sorted_data]

    edge_colors = ['red' if model in models_on_skyline else 'black' for model in models]
    model_colors = [colors[model] for model in models]

    add_margins(axis, 30, acc_metric)

    [axis.scatter(
        footprints[i], accuracies[i], label=PROPER_MODEL_NAMES[models[i]], zorder=5,
        edgecolors=edge_colors[i], color=model_colors[i]
    ) for i in range(len(models))]
    
    proper_names = [f"{PROPER_MODEL_NAMES[model]}" for model in models]

    texts = [axis.text(x, y, name, va='center', ha='center', fontsize=12)
        for x,y,name in zip(footprints, accuracies, proper_names)]

    adjust_text(texts, 
        expand_align=(1.4, 1.7),
        expand_points=(1.5, 1.8),
        arrowprops=dict(arrowstyle="-", color='black', lw=0.5))


def plot_pareto(data, axis):
    points_x = []
    min_y = min(x[1] for x in data)
    min_y = (math.floor(min_y * 100.0)) / 100.0
    points_y = [min_y]
    highest_acc = 0
    models_on_skyline = set()
    for model, accuracy, footprint in data:
        y = highest_acc
        if accuracy > highest_acc:
            highest_acc = accuracy
            y = accuracy
            models_on_skyline.add(model)
        points_x.extend([footprint] * 2)
        points_y.extend([y] * 2)

    points_x.append(points_x[-1])

    axis.grid(b=True, which="major", axis="both", linestyle="--")
    axis.plot(points_x, points_y, linestyle=":", linewidth=1, c="red")
    return models_on_skyline, (points_x[0], points_y[0]), (points_x[-1], points_y[-1])

def plot_results(language, acc_metric, size_metric, save_to_file):
    results = load_results()
    _, ax = plt.subplots(figsize=(8, 6))
    #plt.margins(0)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(ticker.LogLocator(base=10))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))

    lang_desc = f"{language}_stanford" if INCLUDE_STANFORD else language 

    plot_id = f"{lang_desc}_{acc_metric}_{size_metric}"
    data_for_lang = get_data_for_language(results, language, acc_metric, size_metric)
    sorted_data = sort_data_by_size(data_for_lang, acc_metric, size_metric)

    models_on_skyline, start, end = plot_pareto(sorted_data, ax)
    plot_data(sorted_data, models_on_skyline, ax, plot_id, acc_metric, size_metric)
    
    ax.plot([start[0], start[0]], [start[1], ax.get_ylim()[0]], linestyle=":", linewidth=1, c="red") 
    ax.plot([end[0], ax.get_xlim()[1]], [end[1], end[1]], linestyle=":", linewidth=1, c="red")

    #manager = plt.get_current_fig_manager()
    #w, h = 1920, 1080 #manager.window.maxsize()
    #if w > 1920:
    #    w = 1920
    #manager.resize(w, h)

    #fig_w, fig_h = 12, 7
    #fig.set_size_inches(fig_w, fig_h)

    plt.tight_layout()

    if save_to_file:
        lang_desc = language
        avg_desc = ""
        if language == "avg":
            avg_desc = "_with_stanford" if INCLUDE_STANFORD else "_all"

        if not os.path.exists("plots"):
            os.mkdir("plots")

        filename = f"plots/{lang_desc}-{acc_metric}_{size_metric}{avg_desc}.png"
        plt.savefig(filename)
        print(f"Saved image of plot to {filename}.")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot skyline/pareto curve of accuracy vs. size of POS taggers")

    choices_langs = list(LANGS_FULL.keys())
    choices_langs.append("avg")
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
    parser.add_argument(
        "-is", "--include_stanford", action="store_true",
        help="Whether to include plots for Stanford (only relevant when plotting for 'avg' across languages)"
    )

    args = parser.parse_args()
    INCLUDE_STANFORD = args.include_stanford

    plot_results(LANGS_ISO.get(args.language, "avg"), args.accuracy_metric, args.size_metric, args.save)
