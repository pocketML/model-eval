from sys import argv
from os import unlink
import shutil
import requests

URL = "https://magnusjacobsen.com/pocket-ml/models.tar.gz"

def download_and_unpack():
    response = requests.get(URL)

    filename = f"models/{URL.split('/')[-1]}"
    print(f"Downloading models from {URL}...")

    with open(filename, "wb") as fp:
        for chunk in response.iter_content(chunk_size=128):
            fp.write(chunk)

    try:
        shutil.unpack_archive(filename, f"models/")
        unlink(filename) # Remove unpacked archive.
    except ValueError:
        pass

if __name__ == "__main__":
    download_and_unpack()
