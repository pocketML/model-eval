from glob import glob
import matplotlib.pyplot as plt
import numpy as np

def find_value(lines, key):
    for line in lines:
        if key in line:
            split = line.split(":")
            return split[1].strip()
    return None

def load_results():
    result_files = glob("results/plot/*.out")
    mapped_data = {}
    for file_name in result_files:
        last_part = file_name.replace("\\", "/").split("/")[-1]
        model_name = "_".join(last_part.split("_")[:-2])
        if model_name not in mapped_data:
            with open(file_name, "r") as fp:
                lines = fp.readlines()
                token_acc = find_value(lines, "Final token acc")
                sentence_acc = find_value(lines, "Final sentence acc")
                memory_footprint = find_value(lines, "Memory usage")
                code_size = find_value(lines, "Code size")
                model_size = find_value(lines, "Model size")
                compressed_size = find_value(lines, "Compressed size")
                mapped_data[model_name] = {
                    "token": token_acc, "sentence": sentence_acc,
                    "memory": memory_footprint, "code": code_size,
                    "model": model_size, "compressed": compressed_size
                }
    return mapped_data

def pareto_distribution(x, a, L, H):
    cummulative = False
    if cummulative:
        return (1 - L ** a * x ** -a) / (1 - (L/H) ** a)
    return (a * L ** a * x ** (-a - 1)) / (1 - (L/H) ** a)

def plot_data(data, acc_measure="token", size_measure="memory"):
    legend_text = {
        "token": "Token Accuracy", "sentence": "Sentence Accuracy",
        "memory": "Memory Footprint", "code": "Code Size",
        "model": "Uncompresse Model Size", "compressed": "Compressed Model Size"
    }
    plt.xlabel(f"{legend_text[size_measure]} (MB)")
    plt.ylabel(legend_text[acc_measure])

    sorted_data = []
    for model in data:
        model_data = data[model]
        accuracy = float(model_data[acc_measure])
        footprint = int(model_data[size_measure]) / 1000
        sorted_data.append((model, accuracy, footprint))

    sorted_data.sort(key=lambda x: x[2])

    for model, accuracy, footprint in sorted_data:
        x, y = accuracy, footprint
        offset_x = 50
        offset_y = 0.005
        plt.text(x - offset_x, y + offset_y, f"Acc = {x}")
        plt.text(x - offset_x, y + offset_y, f"Footprint = {y} MB")
        plt.scatter(footprint, accuracy, label=model)

    plt.legend()
    return sorted_data

def plot_pareto(data):
    points_x = [min(x[2] for x in data) - 0.1]
    points_y = []
    highest_acc = 0
    for _, accuracy, footprint in data:
        y = highest_acc
        if accuracy > highest_acc:
            highest_acc = accuracy
            y = accuracy
        points_x.extend([footprint] * 2)
        points_y.extend([y] * 2)

    points_y.append(points_y[-1])

    plt.grid(b=True, which="major", axis="both")
    plt.plot(points_x, points_y)

def plot_results():
    results = load_results()
    sorted_data = plot_data(results)
    plot_pareto(sorted_data)
    plt.show()

if __name__ == "__main__":
    plot_results()
