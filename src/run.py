import asyncio
from operator import mod
import os
import shutil
import subprocess
from datetime import datetime
import platform
import argparse
import nltk
import nltk_util
import data_archives
import transform_data
import taggers
from inference import monitor_inference, InferenceTask
from training import monitor_training

if "JAVAHOME" not in os.environ:
    java_exe_path = shutil.which("java")
    if java_exe_path is None:
        print("WARNING: 'JAVAHOME' environment variable not set!")
    else:
        java_path = "/".join(java_exe_path.replace("\\", "/").split("/")[:-1])
        os.environ["JAVAHOME"] = java_path

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

NLTK_MODELS = { # Entries are model_name -> (model_class, args)
    "tnt": (nltk.TnT, []),
    "stanford": (
        nltk.StanfordPOSTagger,
        [
            "D:/model-eval/models/stanford-tagger/models/english-bidirectional-distsim.tagger",
            "D:/model-eval/models/stanford-tagger/stanford-postagger.jar"
        ]
    )
}

# tnt = nltk.TnT()
# hmm = nltk.HiddenMarkovModelTagger()
# senna = nltk.Senna()
# brill = nltk.BrillTagger()
# crf = nltk.CRFTagger()
# stanford = nltk.StanfordPOSTagger()

TAGGERS = {
    "svmtool": taggers.SVMT,
    "bilstm": taggers.BILSTM,
    "pos_adv": taggers.POSADV,
    "stanford": taggers.Stanford
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

async def run_with_sys_call(args, model_name, file_pointer):
    call_train, call_infer = MODELS_SYS_CALLS[model_name]
    tagger = TAGGERS[model_name](args.lang, args.treebank)

    cwd = os.getcwd().replace("\\", "/")
    if not os.path.exists(f"{cwd}/{tagger.model_base_path()}"):
        os.mkdir(f"{cwd}/{tagger.model_base_path()}")

    final_acc = 0
    if args.train: # Train model.
        call_train = insert_arg_values(call_train, tagger, args, model_name)
        process = system_call(call_train, cwd)
        final_acc = await monitor_training(tagger, process, args, file_pointer)

    model_footprint = None
    if args.predict: # Run inference task.
        if call_infer is None:
            call_infer = insert_arg_values(call_infer, tagger, args, model_name)
            process = system_call(call_infer, cwd)
            task = InferenceTask(process, InferenceTask.TASK_SYSCALL)
            model_footprint = await monitor_inference(task)
        final_acc = tagger.get_pred_acc()
    return final_acc, model_footprint

async def run_with_nltk(args, model_name, file_pointer):
    model_class, model_args = NLTK_MODELS[model_name]
    model = model_class(*model_args)

    final_acc = 0
    if args.train: # Train model.
        print(f"Training NLTK model: '{model_name}'")
        train_data = nltk_util.format_nltk_data(args, "train")
        model.train(train_data)

    model_footprint = None
    if args.predict: # Run inference task.
        test_data = nltk_util.format_nltk_data(args, "test")
        asyncio_task = asyncio.create_task(nltk_util.evaluate(model, test_data))
        task = InferenceTask(asyncio_task, InferenceTask.TASK_ASYNCIO)
        model_footprint = await monitor_inference(task)
        final_acc = asyncio_task.result()
    return final_acc, model_footprint

async def main(args):
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

    models_to_run = (list(MODELS_SYS_CALLS.keys()) + list(NLTK_MODELS.keys())
                     if args.model_name == "all" else [args.model_name])
    if not args.train and not args.predict: # Do both training and inference.
        args.train = True
        args.predict = True

    if args.treebank is None: # Get default treebank for given langauge, if none is specified.
        args.treebank = data_archives.get_default_treebank(args.lang)

    for model_name in models_to_run:
        file_pointer = None
        if args.save_results:
            formatted_date = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
            file_name = f"results/{model_name}_{formatted_date}.out"
            file_pointer = open(file_name, "w")

        if model_name in MODELS_SYS_CALLS:
            final_acc, model_footprint = await run_with_sys_call(args, model_name, file_pointer)
        elif model_name in NLTK_MODELS:
            final_acc, model_footprint = await run_with_nltk(args, model_name, file_pointer)

        if model_footprint is not None:
            print(f"Model footprint: {model_footprint}")
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
    print("*****************************************")
    print("*****************************************")
    print("****                                 ****")
    print("***   pocketML model evaluator 4000   ***")
    print("****                                 ****")
    print("*****************************************")
    print("*****************************************\n")

    parser = argparse.ArgumentParser(description="Evaluation of various state of the art POS taggers, on the UD dataset")
    
    # required arguments (positionals)
    choices = list(MODELS_SYS_CALLS.keys()) + list(NLTK_MODELS.keys()) + ["all"]
    parser.add_argument("model_name", type=str, choices=choices, help="name of the model to run")

    # optional arguments
    parser.add_argument("-l", "--lang", type=str, default="en", help="choose dataset language. Default is English.")
    parser.add_argument("-i", "--iter", type=int, default=10, help="number of training iterations. Default is 10.")
    parser.add_argument("-v", "--verbose", help="increase output verbosity")
    parser.add_argument("-tb", "--treebank", type=str, help="UD treebank to use as dataset (fx. 'gum')", default=None, required=False)
    parser.add_argument("-lb", "--loadbar", help="whether to run with loadbar", action="store_true")
    parser.add_argument("-s", "--save-results", help="whether to save accuracy & size complexity measurements", action="store_true")
    parser.add_argument("-t", "--train", help="whether to train the given model", action="store_true")
    parser.add_argument("-p", "--predict", help="whether to predict/evaluate accuracy using the given model", action="store_true")

    args = parser.parse_args()

    asyncio.run(main(args))
