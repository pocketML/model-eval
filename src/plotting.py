from glob import glob
import matplotlib.pyplot as plt
import numpy as np

def load_results():
    result_files = glob("results/*.out")
    mapped_data = {}
    for file_name in result_files:
        last_part = file_name.replace("\\", "/").split("/")[-1]
        model_name = last_part.split("_")[0]
        if model_name not in mapped_data:
            with open(file_name, "r") as fp:
                lines = fp.readlines()
                accuracy = float(lines[-3].split(":")[1].strip())
                footprint = float(lines[-1].split(":")[1].strip())
                mapped_data[model_name] = (accuracy, footprint)
    return mapped_data

def pareto_distribution(x, a, L, H):
    return (a * L ** a * x ** (-a-1)) / (1 - (L/H) ** a)

def plot_pareto(data):
    plt.xlabel("Accuracy")
    plt.ylabel("Memory Footprint (MB)")

    for model in data:
        accuracy, footprint = data[model]
        x, y = accuracy, int(footprint // 1000)
        offset_x = 0.015
        offset_y = 50
        plt.text(x - offset_x, y + offset_y, f"Acc = {x}")
        plt.text(x - offset_x, y + offset_y - 25, f"Footprint = {y} MB")
        plt.scatter(accuracy, footprint // 1000, label=model)

    accuracies = [acc for (acc, _) in data.values()]
    footprints = [mem // 1000 for (_, mem) in data.values()]
    x_pareto = np.linspace(min(accuracies), max(accuracies), 500)
    y_pareto = [max(footprints) - pareto_distribution(x, 5, min(footprints), max(footprints)) for x in x_pareto]
    plt.plot(x_pareto, y_pareto)

    plt.legend()
    plt.show()

def plot_results():
    results = load_results()
    plot_pareto(results)

if __name__ == "__main__":
    plot_results()
