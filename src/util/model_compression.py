import math
import shutil

def compress(model_filename, format="zip"):
    shutil.make_archive(model_filename + "_compressed", format)

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

if __name__ == "__main__":
    filename = "models/bilstm_aux/pocketML/en_gum/pocketML.model"
    compress_format = "zip"
    print(f"Entropy before: {shannon_entropy(filename):.3f}")
    print("Compressing model...")
    compress(filename, compress_format)
    print("File compressed.")
    filename = filename + "." + compress_format
    print(f"Entropy after: {shannon_entropy(filename):.3f}")
