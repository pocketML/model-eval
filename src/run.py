import asyncio
import os
import subprocess
from datetime import datetime
import platform
import argparse
import data_archives
import transform_data
import taggers
from inference import monitor_inference
from training import monitor_training

MODELS_SYS_CALLS = { # Entries are model_name -> (sys_call_train, sys_call_predict)
    "bilstm": (
        (
            "python [dir]/models/bilstm-aux/src/structbilty.py --dynet-mem 1500 " +
            "--train data/UD_English-GUM/simplified/en_gum-ud-train.conllu " +
            "--dev data/UD_English-GUM/simplified/en_gum-ud-dev.conllu " +
            f"--test data/UD_English-GUM/simplified/en_gum-ud-test.conllu --iters [iters] --model {taggers.BILSTM.SAVED_MODEL}"
        ),
        (
            f"python [dir]/models/bilstm-aux/src/structbilty.py --model {taggers.BILSTM.SAVED_MODEL} " +
            "--test data/UD_English-GUM/simplified/en_gum-ud-test.conllu " +
            f"--output {taggers.BILSTM.PREDICTIONS}"
        )
    ),
    "svmtool": (
        "bash -c \"perl [dir]/models/svmtool/bin/SVMTlearn.pl -V 1 models/svmtool/bin/config.svmt\"",
        (f"bash -c \"perl [dir]/models/svmtool/bin/SVMTagger.pl {taggers.SVMT.SAVED_MODEL}/{taggers.SVMT.latest_model()} < " +
         f"data/UD_English-GUM/simplified/en_gum-ud-test.conllu > {taggers.SVMT.PREDICTIONS}\"")
    ),
    "pos_adv": ("cd [dir]/models/pos_adv && multi_lingual_run_blstm-blstm-crf_pos.sh", None),
}

TAGGERS = {
    "svmtool": taggers.SVMT,
    "bilstm": taggers.BILSTM,
    "pos_adv": taggers.POSADV
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
    parser.add_argument("-t", "--train", action="store_true")
    parser.add_argument("-p", "--predict", action="store_true")

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
    do_training = args.train
    do_inference = args.predict
    if not args.train and not args.predict: # Do both training and inference.
        do_training = True
        do_inference = True

    for model_name in models_to_run:
        call_train, call_infer = MODELS_SYS_CALLS[model_name]
        file_pointer = None
        if args.save_results:
            formatted_date = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
            file_name = f"results/{model_name}_{formatted_date}.out"
            file_pointer = open(file_name, "w")

        final_acc = 0
        if do_training: # Traing model.
            process = system_call(call_train, args.iter)
            tagger = TAGGERS[model_name](process)
            final_acc = await monitor_training(tagger, args, file_pointer)

        if do_inference: # Run inference task.
            process = system_call(call_infer, args.iter)
            tagger = TAGGERS[model_name](process)
            model_footprint = await monitor_inference(tagger)
            final_acc = tagger.get_pred_acc()
            if file_pointer is not None: # Save size of model footprint.
                file_pointer.write(f"Model footprint: {model_footprint}\n")

        # Normalize accuracy
        if final_acc > 1:
            final_acc /= 100
        normed_acc = f"{final_acc:.4f}"
        print(f"Test Accuracy: {normed_acc}")
        if file_pointer is not None: # Save final test-set/prediction accuracy.
                file_pointer.write(f"Final acc: {normed_acc}\n")
                file_pointer.close()

if __name__ == "__main__":
    asyncio.run(main())
