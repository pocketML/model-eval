import asyncio
import os
import multiprocessing
import shutil
import subprocess
from datetime import datetime
import platform
import argparse
import nltk
from nltk.tbl.template import Template
from nltk.tag.brill import Word, Pos
import nltk_util
import data_archives
import transform_data
import taggers
import plotting
from inference import monitor_inference
from training import monitor_training, train_nltk_model

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
    ),
    "stanford": (
        (
            "java -cp models/stanford-tagger/stanford-postagger.jar " +
            "edu.stanford.nlp.tagger.maxent.MaxentTagger -props [model_base_path]/pocketML.props"
        ),
        (
            "java -mx2g -cp models/stanford-tagger/stanford-postagger.jar " +
            "edu.stanford.nlp.tagger.maxent.MaxentTagger -model [model_path] " +
            "--encoding UTF-8 " +
            "--testFile format=TSV,wordColumn=0,tagColumn=1,[dataset_test] [stdout] [pred_path]"
        )
    )
}

NLTK_MODELS = { # Entries are model_name -> (model_class, args)
    "tnt": (nltk.TnT, []),
    #"brill": 
    #    (nltk.BrillTaggerTrainer, [
    #        nltk_util.load_model("tnt"),
    #        [
    #            Template(Pos([-1])), Template(Pos([1])), Template(Pos([-2])),
    #            Template(Pos([2])), Template(Word([0])), Template(Word([1, -1]))
    #        ]
    #    ]
    #)
}

# hmm = nltk.HiddenMarkovModelTagger()
# senna = nltk.Senna()
# brill = nltk.BrillTagger()
# crf = nltk.CRFTagger()

TAGGERS = {
    "svmtool": taggers.SVMT,
    "bilstm": taggers.BILSTM,
    "pos_adv": taggers.POSADV,
    "stanford": taggers.Stanford
}

def insert_arg_values(cmd, tagger, args):
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

    stderr_reroute = subprocess.STDOUT
    if "[stderr]" in cmd:
        index = cmd.index("[stderr]")
        file_path = cmd[index + len("[stderr]") + 1:]
        stderr_reroute = open(file_path, "w", encoding="utf-8")
        cmd = cmd[:index]

    stdout_reroute = subprocess.PIPE
    if "[stdout]" in cmd:
        index = cmd.index("[stdout]")
        file_path = cmd[index + len("[stdout]") + 1:]
        stdout_reroute = open(file_path, "w", encoding="utf-8", errors="utf-8")
        cmd = cmd[:index]

    cmd_full = cmd.replace("[dir]", cwd)
    #if platform.system() == "Windows" and not cmd.startswith("bash"):
    #    cmd_full = cmd_full.replace("/", "\\")
    print(f"Running {cmd_full}")
    
    process = subprocess.Popen(cmd_full, stdout=stdout_reroute, stderr=stderr_reroute)
    return process

async def run_with_sys_call(args, model_name, tagger_helper, file_pointer):
    call_train, call_infer = MODELS_SYS_CALLS[model_name]

    cwd = os.getcwd().replace("\\", "/")
    if not os.path.exists(f"{cwd}/{tagger_helper.model_base_path()}"):
        os.mkdir(f"{cwd}/{tagger_helper.model_base_path()}")

    final_acc = 0
    if args.train: # Train model.
        call_train = insert_arg_values(call_train, tagger_helper, args)
        process = system_call(call_train, cwd)
        final_acc = await monitor_training(tagger_helper, process, args, file_pointer)

    model_footprint = None
    if args.eval: # Run inference task.
        if call_infer is not None:
            call_infer = insert_arg_values(call_infer, tagger_helper, args)
            process = system_call(call_infer, cwd)
            model_footprint = await monitor_inference(process)
        final_acc = tagger_helper.get_pred_acc()
    return final_acc, model_footprint

async def run_with_nltk(args, model_name):
    model_class, model_args = NLTK_MODELS[model_name]
    model = model_class(*model_args)

    final_acc = 0
    if args.train: # Train model.
        print(f"Training NLTK model: '{model_name}'")
        train_data = nltk_util.format_nltk_data(args.lang, args.treebank, "train")
        trained_model = train_nltk_model(model, train_data, args)
        nltk_util.save_model(trained_model, model_name)

    model_footprint = None
    if args.eval: # Run inference task.
        if nltk_util.saved_model_exists(model_name):
            model = nltk_util.load_model(model_name)
        test_data = nltk_util.format_nltk_data(args.lang, args.treebank, "test")

        # We run NLTK model inference in a seperate process,
        # so we can measure it's memory usage similarly to a system call.
        pipe_1, pipe_2 = multiprocessing.Pipe()
        process = multiprocessing.Process(target=nltk_util.evaluate, args=(model, test_data, pipe_2))
        process.start()

        # Wait for inference to complete.
        model_footprint = await monitor_inference(process)
        final_acc = pipe_1.recv()
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
    if not args.train and not args.eval: # Do both training and inference.
        args.train = True
        args.eval = True

    if args.treebank is None: # Get default treebank for given langauge, if none is specified.
        args.treebank = data_archives.get_default_treebank(args.lang)

    for model_name in models_to_run:
        file_pointer = None
        if args.save_results:
            if not os.path.exists("results"):
                os.mkdir("results")
            formatted_date = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
            file_name = f"results/{model_name}_{formatted_date}.out"
            file_pointer = open(file_name, "w")

        tagger = TAGGERS.get(model_name, None)
        if tagger is not None: # Instantiate helper class, if it exists relevant.
            tagger = tagger(args)

        if model_name in MODELS_SYS_CALLS:
            acc_tuple, model_footprint = await run_with_sys_call(args, model_name, tagger, file_pointer)
        elif model_name in NLTK_MODELS:
            acc_tuple, model_footprint = await run_with_nltk(args, model_name)

        token_acc, sent_acc = acc_tuple
        # Normalize accuracy
        if token_acc > 1:
            token_acc /= 100
        if sent_acc > 1:
            sent_acc /= 100
        token_acc = f"{token_acc:.4f}"
        sent_acc = f"{sent_acc:.4f}"
        print(f"Token Accuracy: {token_acc}")
        print(f"Sentence Accuracy: {sent_acc}")

        if model_footprint is not None:
            print(f"Model footprint: {model_footprint}KB")

        if file_pointer is not None: # Save final test-set/prediction accuracy.
            file_pointer.write(f"Final token acc: {token_acc}\n")
            file_pointer.write(f"Final sentence acc: {sent_acc}\n")

            if model_footprint is not None: # Save size of model footprint.
                file_pointer.write(f"Model footprint: {model_footprint}\n")

            file_pointer.close()

    if args.plot:
        plotting.plot_results()

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
    parser.add_argument("-e", "--eval", help="whether to predict & evaluate accuracy using the given model", action="store_true")
    parser.add_argument("-p", "--plot", help="whether to plot results from previous/current runs", action="store_true")

    args = parser.parse_args()

    asyncio.run(main(args))
