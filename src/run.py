import asyncio
import os
import subprocess
from datetime import datetime
import platform
import argparse
import data_archives
import transform_data
import tagger_monitors
from inference import monitor_inference
from training import monitor_training

MODELS_SYS_CALLS = {
    "bilstm": (
        "python [dir]/models/bilstm-aux/src/structbilty.py --dynet-mem 1500 " +
        "--train data/UD_English-GUM/simplified/en_gum-ud-train.conllu " +
        "--dev data/UD_English-GUM/simplified/en_gum-ud-dev.conllu " +
        "--test data/UD_English-GUM/simplified/en_gum-ud-test.conllu --iters [iters] --model models/bilstm-aux/en"
    ),
    "svmtool": "bash -c \"perl [dir]/models/svmtool/bin/SVMTlearn.pl -V 1 models/svmtool/bin/config.svmt\"",
    "svmtool_tag": "bash -c \"perl [dir]/models/svmtool/bin/SVMTagger.pl models/svmtool/pocketML/pocketML.FLD.8 < data/UD_English-GUM/simplified/en_gum-ud-test.conllu > models/svmtool/eng_gum.out\"",
    "svmtool_eval": "bash -c \"perl [dir]/models/svmtool/bin/SVMTeval.pl 0 models/svmtool/pocketML/pocketML.FLD.8 data/UD_English-GUM/simplified/en_gum-ud-test.conllu models/svmtool/eng_gum.out\"",
    "pos_adv": "cd [dir]/models/pos_adv && ./multi_lingual_run_blstm-blstm-crf_pos.sh",
    "test": "python [dir]/src/test.py"
}

OUTPUT_MONITORS = {
    "svmtool": tagger_monitors.SVMTParser,
    "bilstm": tagger_monitors.BILSTMParser,
    "test": tagger_monitors.BILSTMParser
}

def system_call(cmd, iters):
    curr_dir = os.getcwd()
    if platform.system() == "Windows" and cmd.startswith("bash"):
        print("Running in Linux shell")
        curr_dir = "/" + curr_dir.replace("\\", "/")
    cmd_full = cmd.replace("[dir]", curr_dir).replace("[iters]", str(iters))
    if platform.system() == "Windows" and not cmd.startswith("bash"):
        cmd_full = cmd_full.replace("/", "\\")
    print(f"Running {cmd_full}")
    process = subprocess.Popen(cmd_full, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return process

async def main():
    print("*****************************************")
    print("*****************************************")
    print("****                                 ****")
    print("***   pocketML model evaluator 4000   ***")
    print("****                                 ****")
    print("*****************************************")
    print("*****************************************\n")

    parser = argparse.ArgumentParser(description="Evaluation of various state of the art POS taggers, on the UD dataset")
    
    # required arguments (positionals)
    choices = list(MODELS_SYS_CALLS.keys()) + ["all"]
    parser.add_argument("model_name", type=str, choices=choices, help="name of the model to run")

    # optional arguments
    parser.add_argument("-l", "--lang", type=str, default="en", help="choose dataset language. Default is English.")
    parser.add_argument("-i", "--iter", type=int, default=10, help="number of training iterations. Default is 10.")
    parser.add_argument("-v", "--verbose", help="increase output verbosity")
    parser.add_argument("-nl", "--no-loadbar", help="run with no loadbar", action="store_true")
    parser.add_argument("-s", "--save-results", help="save accuracy/size complexity measurements", action="store_true")

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

    models_to_run = MODELS_SYS_CALLS.keys() if args.model_name == "all" else [args.model_name]

    for model_name in models_to_run:
        sys_call = MODELS_SYS_CALLS[args.model_name]
        file = None
        if args.save_results:
            formatted_date = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
            file_name = f"results/{args.model_name}_{formatted_date}.out"
            file = open(file_name, "w")

        process = system_call(sys_call, args.iter)
        monitor = OUTPUT_MONITORS[args.model_name](process)
        final_acc = await monitor_training(monitor, args, file)
        file.write("Final acc: {final_acc}")
        file.close()

if __name__ == "__main__":
    asyncio.run(main())
