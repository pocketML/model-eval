import math
from taggers import stanford

def shannon_entropy(model_filename):
    with open(model_filename, "rb") as fp:
        byte_arr = fp.read()
        counts = [0] * 256
        entropy = 0.0
        length = len(byte_arr)

        readable_bytes = length
        descriptor = "Bytes"
        if (readable_bytes > 1000):
            readable_bytes /= 1000
            descriptor = "KB"
        if (readable_bytes > 1000):
            readable_bytes /= 1000
            descriptor = "MB"

        print(
            "Calculating Shannon entropy on file with a size of " +
            f"{readable_bytes:.2f}{descriptor}"
        )

        for byte in byte_arr:
            counts[byte] += 1

        for count in counts:
            if count != 0:
                probability = float(count) / length
                entropy -= probability * math.log(probability, 2)

        return entropy

def find_best_compression_method(tagger):
    formats = ["zip", "gztar", "bztar", "xztar"]
    exts = ["zip", "tar.gz", "tar.bz2", "tar.xz"]
    results = []
    for comp_format, ext in zip(formats, exts):
        entropy_before = shannon_entropy(tagger.model_path())
        size_before = tagger.model_size()
        print(f"Entropy before: {entropy_before}")
        print(f"Size before: {size_before}")
        print("Compressing...")
        compressed = tagger.compress_model(comp_format)
        entropy_after = shannon_entropy(compressed)
        size_after = tagger.compressed_model_size(ext)
        results.append(
            (comp_format, entropy_after - entropy_before, size_before - size_after)
        )
        print(f"Entropy after: {entropy_after}")
        print(f"Size after: {size_after}")

    return sorted(results, key=lambda x: x[1], reverse=True)

if __name__ == "__main__":
    class Args:
        def __init__(self):
            self.lang = "en"
            self.treebank = "gum"
            self.iter = 10

    best_format = find_best_compression_method(stanford.Stanford(Args(), "stanford"))
    print(best_format)
