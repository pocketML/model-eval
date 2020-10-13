from glob import glob
from os import unlink
from shutil import unpack_archive
import requests

def download_and_unpack(archive):
    url = f"https://magnusjacobsen.com/projects/pocketml/{archive}.tgz"
    response = requests.get(url)

    filename = f"{archive}/{url.split('/')[-1]}"
    print(f"Downloading {archive} from {url}...")

    with open(filename, "wb") as fp:
        for chunk in response.iter_content(chunk_size=128):
            fp.write(chunk)

    try:
        unpack_archive(filename, f"{archive}/")
        unlink(filename) # Remove old archive.
    except ValueError:
        pass

def archive_exists(archive):
    return len(glob(archive + "/*")) > 1

if __name__ == "__main__":
    download_and_unpack("models")
