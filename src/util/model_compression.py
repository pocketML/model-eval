import math

COMPRESSION_EXTS = {
    "zip": "zip",
    "gztar": "tar.gz",
    "bztar": "tar.bz2",
    "xztar": "tar.xz"
}

def shannon_entropy(*filenames):
    """
    Calculate Shannon Entropy for a variable set of files.
    """
    total_length = 0
    counts = [0] * 256
    entropy = 0.0

    for filename in filenames:
        with open(filename, "rb") as fp:
            byte_arr = fp.read() # Read bytes.
            total_length += len(byte_arr) # Add length of bytes to total length.

            for byte in byte_arr: # Save frequencies of each byte.
                counts[byte] += 1

    for count in counts:
        if count != 0:
            probability = float(count) / total_length
            entropy -= probability * math.log(probability, 2)

    return entropy
