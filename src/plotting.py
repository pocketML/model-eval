from glob import glob
import matplotlib.pyplot as plt

def load_results():
    result_files = glob("results/*.out")
    mapped_data = {}
    for file_name in result_files:
        last_part = file_name.replace("\\", "/").split("/")[-1]
        model_name = last_part.split("_")[0]
        with open(file_name, "r") as fp:
            lines = fp.readlines()
            accuracy = float(lines[-2].split(":")[1].strip())
            footprint = float(lines[-1].split(":")[1].strip())
            mapped_data[model_name] = (accuracy, footprint)
    return mapped_data

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

    plt.legend()
    plt.show()

def plot_results():
    results = load_results()
    plot_pareto(results)

if __name__ == "__main__":
    plot_results()
