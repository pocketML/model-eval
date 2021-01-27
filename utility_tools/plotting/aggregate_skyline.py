import math
import numpy as np
import read_data
import matplotlib
import matplotlib.pyplot as plt
import read_data

def get_models_on_skyline(data):
    highest_acc = 0
    models_on_skyline = set()
    for model, accuracy, _ in data:
        if accuracy > highest_acc:
            highest_acc = accuracy
            models_on_skyline.add(model)
    return models_on_skyline

def get_skyline_count(acc_metric, size_metric, include_stanford=True):
    languages, taggers, acc =  read_data.get_data(acc_metric, include_stanford=include_stanford)
    size =  read_data.get_data(size_metric, include_stanford=include_stanford)[2]

    taggers_by_language = np.repeat(np.array(taggers), len(languages)).reshape((len(taggers), len(languages))).T
    acc_by_language = np.array(acc).T
    size_by_language = np.array(size).T

    skyline_count = {x: 0 for x in taggers}
    for taggers, accs, sizes in zip(taggers_by_language, acc_by_language, size_by_language):
        zipped = list(zip(taggers, accs, sizes))
        zipped.sort(key=lambda x: x[2])
        models_on_skyline = get_models_on_skyline(zipped)
        for model in models_on_skyline:
            skyline_count[model] += 1

    return taggers, list(skyline_count.values())

def plot(xlabels, bar_labels, data, ylabel, title, yscale, yticks):
    x = np.arange(len(xlabels))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots()
    n = len(bar_labels)
    all_rects = []
    offsets = [-width - width/2, -width/2, width/2, width + width/2]
    for i in range(n):
        rects = ax.bar(x + offsets[i], data[i], width, label=bar_labels[i].capitalize(), zorder=3)
        all_rects.append(rects)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(ylabel)
    plt.yscale(yscale)
    plt.yticks(yticks)
    ax.grid(which='major', axis='y', linestyle='--', zorder=0)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    plt.xticks(rotation=35, ha='right')
    ax.legend()

    fig.tight_layout()

    plt.show()

def aggregate_plotting(acc_metric="token"):
    metrics = ["memory", "code", "model", "compressed"]
    metric_labels = ["Memory", "Code (estimate)", "Model", "Model Compressed"]
    
    results = []
    taggers = None

    for metric in metrics:
        taggers, skyline_counts = get_skyline_count(acc_metric, metric)
        print(metric)
        print(skyline_counts)
        results.append(skyline_counts)

    title = f'On efficiency skyline - {acc_metric} accuracy vs size metric'
    ticks = [i for i in range(11)]
    plot(taggers, metric_labels, results, 'Efficiency count', title, 'linear', ticks)


if __name__ == '__main__':
    aggregate_plotting(acc_metric="sentence")
