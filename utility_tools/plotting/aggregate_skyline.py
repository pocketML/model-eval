import numpy as np
import read_data
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

def plot(xlabels, bar_labels, data, ylabel, title, yscale, yticks, acc_metric):
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

def separate_plots(xlabels, bar_labels, data, ylabel, title, yscale, yticks, acc_metric):
    x = np.arange(len(xlabels))  # the label locations
    width = 0.4  # the width of the bars
    fig, axs = plt.subplots(2,2, figsize=(12, 9))
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

    for i, ax in enumerate(axs.flat):
        ax.bar(x, data[i], width, zorder=3, color=colors[i])

        ax.set_title(f'{acc_metric.capitalize()} accuracy vs. {bar_labels[i].capitalize()}')

        ax.set_ylabel(ylabel)
        ax.set_yscale(yscale)
        ax.set_yticks(yticks)
        ax.grid(which='major', axis='y', linestyle='--', zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels)
        #ax.set_xticks(rotation=35, ha='right')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    fig.tight_layout()
    plt.show()

def aggregate_plotting(acc_metric="token", separate=False):
    metrics = ["memory", "code", "model", "compressed"]
    metric_labels = ["Memory", "Code Size", "Model Size", "Model Compressed"]

    results = []
    sort_order = []

    for metric in metrics:
        taggers, skyline_counts = get_skyline_count(acc_metric, metric)
        if sort_order == []:
            sorted_data = sorted(zip(taggers, skyline_counts), key=lambda x: x[1])
            sort_order = [x[0] for x in sorted_data]
        sorted_data = sorted(zip(taggers, skyline_counts), key=lambda x: sort_order.index(x[0]))
        results.append([x[1] for x in sorted_data])

    title = f'On efficiency skyline - {acc_metric} accuracy vs size metric'
    ticks = [i for i in range(11)]
    if separate:
        separate_plots(sort_order, metric_labels, results, 'Efficiency count', title, 'linear', ticks, acc_metric)
    else:
        plot(sort_order, metric_labels, results, 'Efficiency count', title, 'linear', ticks, acc_metric)

if __name__ == '__main__':
    aggregate_plotting(acc_metric="sentence", separate=True)
