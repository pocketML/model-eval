import asyncio
import os
import subprocess
import platform
import shutil
import argparse
import data_archives
import transform_data
import tracemalloc
import tagger_monitors
import loadbar

MODELS_SYS_CALLS = {
    "bilstm": (
        "python [dir]/models/bilstm-aux/src/structbilty.py --dynet-mem 1500 " +
        "--train data/UD_English-GUM/simplified/en_gum-ud-train.conllu " +
        "--test data/UD_English-GUM/simplified/en_gum-ud-test.conllu --iters 10 --model en"
    ),
    "svmtool": "bash -c \"perl [dir]/models/svmtool/bin/SVMTlearn.pl -V 1 models/svmtool/bin/config.svmt\"",
    "svmtool_tag": "bash -c \"perl [dir]/models/svmtool/bin/SVMTagger.pl models/svmtool/pocketML/pocketML.FLD.8 < data/UD_English-GUM/simplified/en_gum-ud-test.conllu > models/svmtool/eng_gum.out\"",
    "svmtool_eval": "bash -c \"perl [dir]/models/svmtool/bin/SVMTeval.pl 0 models/svmtool/pocketML/pocketML.FLD.8 data/UD_English-GUM/simplified/en_gum-ud-test.conllu models/svmtool/eng_gum.out\"",
    "pos_adv": "cd [dir]/models/pos_adv && ./multi_lingual_run_blstm-blstm-crf_pos.sh"
}

OUTPUT_MONITORS = {
    "svmtool": tagger_monitors.SVMTParser,
    "bilstm": tagger_monitors.BILSTMParser
}

def system_call(cmd):
    tracemalloc.start()
    curr_dir = os.getcwd()
    if platform.system() == "Windows" and cmd.startswith("bash"):
        print("Running in Linux shell")
        curr_dir = "/" + curr_dir.replace("\\", "/")
    cmd_full = cmd.replace("[dir]", curr_dir)
    if platform.system() == "Windows" and not cmd.startswith("bash"):
        cmd_full = cmd_full.replace("/", "\\")
    print(f"Running {cmd_full}")
    pipe = subprocess.Popen(cmd_full, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout
    return pipe

async def monitor_training(monitor, model_name, use_loadbar=True):
    if use_loadbar:
        loadbar_obj = loadbar.Loadbar(30, 100, f"Training ({model_name})")
        loadbar_obj.print_bar()
    async for test_acc in monitor.on_epoch_complete():
        if use_loadbar:
            loadbar_obj.step(text=f"Acc: {test_acc:.2f}%")
        else:
            print(f"Test accuracy: {test_acc}", flush=True)
    print("Current: %d, Peak %d" % tracemalloc.get_traced_memory())

def main():
    print("*****************************************")
    print("*****************************************")
    print("****                                 ****")
    print("***   pocketML model evaluator 4000   ***")
    print("****                                 ****")
    print("*****************************************")
    print("*****************************************\n")

    parser = argparse.ArgumentParser(description="Evaluation of various state of the art POS taggers, on the UD dataset")
    
    # required arguments (positionals)
    parser.add_argument("model_name", type=str, choices=MODELS_SYS_CALLS.keys(), help="name of the model to run")

    # optional arguments
    parser.add_argument("-l", "--lang", type=str, default="en", help="choose dataset language. Default is English.")
    parser.add_argument("-i", "--iter", type=int, default=10, help="number of training iterations. Default is 10.")
    parser.add_argument("-v", "--verbose", help="increase output verbosity")
    parser.add_argument("-nl", "--no-loadbar", help="run with no loadbar", action='store_true')

    args = parser.parse_args()
    print("Arguments:")
    print(f"model: {args.model_name}")
    print(f"verbos: {args.verbose}")
    print(f"dataset language: {args.lang}")
    print(f"iterations: {args.iter}")

    if not data_archives.archive_exists("data"):
        data_archives.download_and_unpack("data")
        transform_data.transform_datasets()
    if not data_archives.archive_exists("models"):
        data_archives.download_and_unpack("models")

    sys_call = MODELS_SYS_CALLS[args.model_name]

    print()

    pipe = system_call(sys_call)
    monitor = OUTPUT_MONITORS[args.model_name](pipe)
    asyncio.run(monitor_training(monitor, args.model_name, not args.no_loadbar))

if __name__ == "__main__":
    main()
