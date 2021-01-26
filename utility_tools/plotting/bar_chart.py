import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import read_data

def plot(labels, metrics, data, ylabel, title, yscale, yticks):
    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots()
    n = len(metrics)
    all_rects = []
    offsets = [-width - width/2, -width/2, width/2, width + width/2]
    for i in range(n):
        rects = ax.bar(x + offsets[i], data[i], width, label=metrics[i], zorder=3)
        all_rects.append(rects)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(ylabel)
    plt.yscale(yscale)
    plt.yticks(yticks)
    #plt.grid(color='grey', which='major', axis='y', linestyle='solid')
    ax.grid(which='major', axis='y', linestyle='--', zorder=0)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(taggers)
    plt.xticks(rotation=35, ha='right')
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{:.2e}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 4, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    #for rects in all_rects:
    #    autolabel(rects)

    fig.tight_layout()

    plt.show()

if __name__ == '__main__':
    metrics, taggers, results = read_data.get_size_data()
    results = np.array(results).T.tolist()
    title = 'Average size measurements for all taggers'
    ticks = [10**i for i in range(9)]
    plot(taggers, metrics, results, 'Kilobytes', title, 'log', ticks)