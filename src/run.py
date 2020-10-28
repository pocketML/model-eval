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
            "--train [dataset_train] " +
            "--dev [dataset_dev] " +
            f"--test [dataset_test] --iters [iters] --model [model_path]"
        ),
        (
            f"python [dir]/models/bilstm-aux/src/structbilty.py --model [model_path] " +
            "--test [dataset_test] " +
            f"--output [pred_path]"
        )
    ),
    "svmtool": (
        "bash -c \"perl [dir]/models/svmtool/bin/SVMTlearn.pl -V 1 models/svmtool/bin/config.svmt\"",
        (f"bash -c \"perl [dir]/models/svmtool/bin/SVMTagger.pl [model_path] < " +
         f"[dataset_test] > [pred_path]\"")
    ),
    "pos_adv": (
        (
            "python [dir]/models/pos_adv/bilstm_bilstm_crf.py --fine_tune --embedding polyglot --oov embedding --update momentum --adv 0.05 " +
            "--batch_size 10 --num_units 150 --num_filters 50 --learning_rate 0.01 --decay_rate 0.05 --grad_clipping 5 --regular none --dropout " +
            "--train [dataset_train] " +
            "--dev [dataset_dev] " +
            "--test [dataset_test] " +
            "--embedding_dict [dir]/models/pos_adv/dataset/word_vec/polyglot-[lang].pkl " +
            f"--output_prediction --patience 30 --exp_dir [model_base_path]"
        ),
        None
    )
}

TAGGERS = {
    "svmtool": taggers.SVMT,
    "bilstm": taggers.BILSTM,
    "pos_adv": taggers.POSADV
}

def insert_arg_values(cmd, tagger, args, model_name):
    replaced = cmd.replace("[iters]", str(args.iter))
    model_base_path = tagger.model_base_path()
    replaced = replaced.replace("[model_base_path]", model_base_path)
    model_path = tagger.model_path()
    replaced = replaced.replace("[model_path]", model_path)
    predict_path = tagger.predict_path()
    replaced = replaced.replace("[pred_path]", predict_path)
    replaced = replaced.replace("[lang]", args.lang)
    dataset_train = data_archives.get_dataset_path(args.lang, args.treebank, "train")
    replaced = replaced.replace("[dataset_train]", dataset_train)
    dataset_test = data_archives.get_dataset_path(args.lang, args.treebank, "test")
    replaced = replaced.replace("[dataset_test]", dataset_test)
    dataset_dev = data_archives.get_dataset_path(args.lang, args.treebank, "dev")
    replaced = replaced.replace("[dataset_dev]", dataset_dev)
    return replaced

def system_call(cmd, cwd):
    if platform.system() == "Windows" and cmd.startswith("bash"):
        split = cwd.replace(" ", "\ ").split("/")
        cwd = "/mnt/" + split[0][:-1].lower() + "/" + "/".join(split[1:])

    cmd_full = cmd.replace("[dir]", cwd)
    #if platform.system() == "Windows" and not cmd.startswith("bash"):
    #    cmd_full = cmd_full.replace("/", "\\")
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
    parser.add_argument("-tb", "--treebank", type=str, help="UD treebank to use as dataset", default=None, required=False)
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

    if args.treebank is None: # Get default treebank for given langauge, if none is specified.
        args.treebank = data_archives.get_default_treebank(args.lang)

    for model_name in models_to_run:
        call_train, call_infer = MODELS_SYS_CALLS[model_name]
        file_pointer = None
        if args.save_results:
            formatted_date = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
            file_name = f"results/{model_name}_{formatted_date}.out"
            file_pointer = open(file_name, "w")

        tagger = TAGGERS[model_name](args.lang, args.treebank)

        cwd = os.getcwd().replace("\\", "/")
        if not os.path.exists(f"{cwd}/{tagger.model_base_path()}"):
            os.mkdir(f"{cwd}/{tagger.model_base_path()}")

        final_acc = 0
        if do_training: # Traing model.
            call_train = insert_arg_values(call_train, tagger, args, model_name)
            process = system_call(call_train, cwd)
            final_acc = await monitor_training(tagger, process, args, file_pointer)

        if do_inference: # Run inference task.
            call_infer = insert_arg_values(call_infer, tagger, args, model_name)
            process = system_call(call_infer, cwd)
            model_footprint = await monitor_inference(tagger, process)
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
