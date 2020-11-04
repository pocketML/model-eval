import asyncio
import os
import multiprocessing
import subprocess
from datetime import datetime
import platform
import argparse
import nltk_util
import data_archives
import plotting
from inference import monitor_inference
from training import monitor_training, train_nltk_model
from taggers import bilstm_aux, bilstm_crf, svmtool, stanford
from taggers import nltk_tnt, nltk_crf, nltk_brill

# hmm = nltk.HiddenMarkovModelTagger()
# senna = nltk.Senna()
# brill = nltk.BrillTagger()
# crf = nltk.CRFTagger()

TAGGERS = {
    "bilstm_aux": bilstm_aux.BILSTMAUX,
    "bilstm_crf": bilstm_crf.BILSTMCRF,
    "svmtool": svmtool.SVMT,
    "stanford": stanford.Stanford,
    "tnt": nltk_tnt.TnT,
    "brill": nltk_brill.Brill,
    "crf": nltk_crf.CRF,
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
    embeddings = data_archives.get_embeddings_path(args.lang)
    replaced = replaced.replace("[embeddings]", embeddings)
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

    if platform.system() != "Windows":
        cmd_full = cmd_full.split(" ")

    process = subprocess.Popen(cmd_full, stdout=stdout_reroute, stderr=stderr_reroute)
    return process

async def run_with_sys_call(args, tagger_helper, file_pointer):
    call_train = tagger_helper.train_string()
    call_infer = tagger_helper.predict_string()

    cwd = os.getcwd().replace("\\", "/")

    final_acc = (0, 0)
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
        final_acc = tagger_helper.evaluate()
    return final_acc, model_footprint

async def run_with_nltk(args, tagger, model_name):
    final_acc = (0, 0)
    if args.train: # Train model.
        print(f"Training NLTK model: '{model_name}'")
        train_data = nltk_util.format_nltk_data(args.lang, args.treebank, "train")
        train_nltk_model(tagger, train_data, args)
        tagger.save_model()

    model_footprint = None
    if args.eval: # Run inference task.
        if not args.train and tagger.saved_model_exists():
            # If we haven't just trained a model, load one for prediction.
            tagger.load_model()
        elif not args.train:
            print("Error: No trained model to predict on!")
            exit(1)

        test_data = nltk_util.format_nltk_data(args.lang, args.treebank, "test")

        # We run NLTK model inference in a seperate process,
        # so we can measure it's memory usage similarly to a system call.
        pipe_1, pipe_2 = multiprocessing.Pipe()
        process = multiprocessing.Process(target=tagger.evaluate, args=(test_data, pipe_2))
        process.start()

        # Wait for inference to complete.
        model_footprint = await monitor_inference(process)
        final_acc = pipe_1.recv() # Receive accuracy from seperate process.
    return final_acc, model_footprint

async def main(args):
    print("Arguments:")
    print(f"model: {args.model_name}")
    print(f"verbose: {args.verbose}")
    print(f"dataset language: {args.lang}")
    print(f"iterations: {args.iter}")

    models_to_run = (TAGGERS.keys()
                     if args.model_name == "all" else [args.model_name])

    for model_name in models_to_run:
        if not TAGGERS[model_name].IS_NLTK:
            if not data_archives.archive_exists("models", model_name):
                data_archives.download_and_unpack("models", model_name)

    language_full = data_archives.LANGUAGES[args.lang]
    if not data_archives.archive_exists("data", language_full):
        data_archives.download_and_unpack("data", language_full)
        data_archives.transform_dataset(language_full)

    if not args.train and not args.eval: # Do both training and inference.
        args.train = True
        args.eval = True

    if args.treebank is None: # Get default treebank for given langauge, if none is specified.
        args.treebank = data_archives.get_default_treebank(args.lang)

    for model_name in models_to_run:
        print(f"Using '{model_name}' model")
        file_pointer = None
        if args.save_results:
            if not os.path.exists("results"):
                os.mkdir("results")
            formatted_date = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
            file_name = f"results/{model_name}_{formatted_date}.out"
            file_pointer = open(file_name, "w")

        tagger = TAGGERS[model_name](args, model_name)

        if tagger.IS_NLTK:
            acc_tuple, model_footprint = await run_with_nltk(args, tagger, model_name)
        else:
            acc_tuple, model_footprint = await run_with_sys_call(args, tagger, file_pointer)

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
    choices_models = list(TAGGERS.keys()) + ["all"]
    parser.add_argument("model_name", type=str, choices=choices_models, help="name of the model to run")

    choices_langs = data_archives.LANGUAGES.keys() - set(data_archives.LANGUAGES.values())

    # optional arguments
    parser.add_argument("-l", "--lang", type=str, choices=choices_langs, default="en", help="choose dataset language. Default is English.")
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
