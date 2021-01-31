from plotting.read_data import get_average_data

def format_size(size):
    scientific = f"{size:.4e}"
    split = scientific.split("e+")
    mantissa = split[0]
    exponent = split[1]
    mantissa_fmt = f"{float(mantissa):.2f}"
    return f"{mantissa_fmt}e{exponent}"

def create_avg_size_file():
    metrics = ["memory", "code", "model", "compressed"]
    results_for_tagger = {}
    for metric in metrics:
        _, taggers, sizes = get_average_data(metric, include_stanford=True)
        for index in range(len(taggers)):
            tagger = taggers[index]
            size = sizes[0][index] if tagger == "Stanford Tagger" else sizes[1][index]
            if tagger not in results_for_tagger:
                results_for_tagger[tagger] = []
            results_for_tagger[tagger].append(format_size(size))

    with open(f"../results_csv/avg_size.csv", "w", encoding="utf-8") as fp:
        fp.write(",".join(metrics) + "\n")
        for tagger in results_for_tagger:
            fp.write(f"{tagger},{','.join(results_for_tagger[tagger])}\n")

if __name__ == "__main__":
    create_avg_size_file()
