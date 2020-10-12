from sys import argv
from os import unlink
import shutil
import requests

URL_SVM_TOOL = "https://www.cs.upc.edu/~nlp/SVMTool/SVMTool++_v_1.1.4.tar.gz"

LINK_DICT = {
    "svmtool": URL_SVM_TOOL
}

if len(argv) <= 1:
    print("Error: Specify which model to download. Options include:")
    for model_name in LINK_DICT:
        print(" - " + model_name)
    print("Or simply 'all' to download all models.")
    exit(0)

response = requests.get(URL_SVM_TOOL)

models_to_load = (LINK_DICT.values()
                  if argv[1] == "all"
                  else (LINK_DICT[arg] for arg in argv[1:]))

for name, link in zip(argv[1:], models_to_load):
    filename = f"models/{link.split('/')[-1]}"
    print(f"Downloading {link}...")
    with open(filename, "wb") as fp:
        for chunk in response.iter_content(chunk_size=128):
            fp.write(chunk)

    try:
        shutil.unpack_archive(filename, f"models/{name}")
        unlink(filename) # Remove unpacked archive.
    except ValueError:
        pass
