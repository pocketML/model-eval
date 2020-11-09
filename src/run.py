import asyncio
import os
import multiprocessing
import subprocess
from datetime import datetime
import platform
import argparse
import data_archives
import plotting
from inference import monitor_inference
from training import monitor_training, train_imported_model
from taggers import bilstm_aux, bilstm_crf, svmtool, stanford, meta_tagger, flair_pos
from taggers import nltk_tnt, nltk_crf, nltk_brill, nltk_hmm

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
    "hmm": nltk_hmm.HMM,
    "meta_tagger": meta_tagger.METATAGGER,
    "flair": flair_pos.Flair,
}

def insert_arg_values(cmd, tagger, args):
    replaced = cmd.replace("[iters]", str(args.iter))
    model_base_path = tagger.model_base_path()
    replaced = replaced.replace("[model_base_path]", model_base_path)
    model_path = tagger.model_path()
    replaced = replaced.replace("[model_path]", model_path)
    predict_path = tagger.predict_path()
    replaced = replaced.replace("[pred_path]", predict_path)
    reload_str = tagger.reload_string()
    if reload_str is None or not args.reload:
        reload_str = ""
    replaced = replaced.replace("[reload]", reload_str)
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

def system_call(cmd, cwd, script_location):
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

    script_path = f"{cwd}/{script_location}"
    if platform.system() == "Windows":
        script_path = f"\"{script_path}\"" 
    cmd_full = cmd.replace("[script_path]", script_path)
    #if platform.system() == "Windows" and not cmd.startswith("bash"):
    #    cmd_full = cmd_full.replace("/", "\\")
    print(f"Running {cmd_full}")

    if platform.system() != "Windows":
        cmd_full = cmd_full.split(" ")

    process = subprocess.Popen(cmd_full, stdout=stdout_reroute, stderr=stderr_reroute)
    return process

async def run_with_sys_call(args, tagger_helper, model_name, file_pointer):
    call_train = tagger_helper.train_string()
    call_infer = tagger_helper.predict_string()

    cwd = os.getcwd().replace("\\", "/")

    final_acc = (0, 0)
    if args.train: # Train model.
        call_train = insert_arg_values(call_train, tagger_helper, args)
        process = system_call(call_train, cwd, tagger_helper.script_path())
        final_acc = await monitor_training(tagger_helper, process, args, model_name, file_pointer)

    model_footprint = None
    if args.eval: # Run inference task.
        if call_infer is not None:
            call_infer = insert_arg_values(call_infer, tagger_helper, args)
            process = system_call(call_infer, cwd, tagger_helper.script_path())
            model_footprint = await monitor_inference(process)
        final_acc = tagger_helper.evaluate()
    return final_acc, model_footprint

async def run_with_imported_model(args, tagger, model_name):
    final_acc = (0, 0)
    if args.train: # Train model.
        print(f"Training imported model: '{model_name}'")
        train_data = tagger.format_data("train")
        train_imported_model(tagger, train_data)
        tagger.save_model()

    model_footprint = None
    if args.eval: # Run inference task.
        test_data = tagger.format_data("test")

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
    print(f"models: {args.model_names}")
    print(f"verbose: {args.verbose}")
    print(f"dataset languages: {args.langs}")
    print(f"iterations: {args.iter}")

    models_to_run = (TAGGERS.keys()
                     if args.model_names == ["all"] else args.model_names)

    for model_name in models_to_run:
        if not TAGGERS[model_name].IS_IMPORTED:
            if not data_archives.archive_exists("models", model_name):
                data_archives.download_and_unpack("models", model_name)

    languages_to_use = (set(data_archives.LANGUAGES.values())
                        if args.langs == ["all"] else args.langs)

    for lang in languages_to_use:
        language_full = data_archives.LANGUAGES[lang]
        if not data_archives.archive_exists("data", language_full):
            data_archives.download_and_unpack("data", language_full)
            data_archives.transform_dataset(language_full)

    if not args.train and not args.eval: # Do both training and inference.
        args.train = True
        args.eval = True

    if len(args.tag) > 0 and len(args.model_names) == 1:
        # Predict POS tags given a tagger and print results.
        args.lang = args.langs[0]
        tagger = TAGGERS[args.model_names[0]](args, args.model_names[0], True)
        tagged_sent = tagger.predict(args.tag)
        print(tagged_sent)
        return

    for model_name in models_to_run:
        for lang in languages_to_use:
            if args.treebank is None: # Get default treebank for given langauge, if none is specified.
                args.treebank = data_archives.get_default_treebank(lang)
            print(
                f"Using '{model_name}' with '{data_archives.LANGUAGES[lang]}' "
                f"dataset on '{args.treebank}' treebank."
            )
            args.lang = lang
            file_pointer = None
            if args.save_results:
                if not os.path.exists("results"):
                    os.mkdir("results")
                formatted_date = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
                file_name = f"results/{model_name}_{formatted_date}.out"
                file_pointer = open(file_name, "w")

            # Load model immediately if we are only evaluating, or if we are continuing training.
            load_model = (args.eval and not args.train) or (args.reload and args.train)
            tagger = TAGGERS[model_name](args, model_name, load_model)

            if tagger.IS_IMPORTED:
                acc_tuple, model_footprint = await run_with_imported_model(args, tagger, model_name)
            else:
                acc_tuple, model_footprint = await run_with_sys_call(args, tagger, model_name, file_pointer)

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
    parser.add_argument("model_names", type=str, choices=choices_models, nargs="+", help="name of the model to run")

    choices_langs = data_archives.LANGUAGES.keys() - set(data_archives.LANGUAGES.values())

    # optional arguments
    parser.add_argument("-tg", "--tag", nargs="+", default=[])
    parser.add_argument("-l", "--langs", type=str, choices=choices_langs, nargs="+", default=["en"], help="choose dataset language(s). Default is English.")
    parser.add_argument("-i", "--iter", type=int, default=10, help="number of training iterations. Default is 10.")
    parser.add_argument("-v", "--verbose", help="increase output verbosity")
    parser.add_argument("-tb", "--treebank", type=str, help="UD treebank to use as dataset (fx. 'gum')", default=None, required=False)
    parser.add_argument("-r", "--reload", help="whether to load a saved model for further training", action="store_true")
    parser.add_argument("-lb", "--loadbar", help="whether to run with loadbar", action="store_true")
    parser.add_argument("-s", "--save-results", help="whether to save accuracy & size complexity measurements", action="store_true")
    parser.add_argument("-t", "--train", help="whether to train the given model", action="store_true")
    parser.add_argument("-e", "--eval", help="whether to predict & evaluate accuracy using the given model", action="store_true")
    parser.add_argument("-p", "--plot", help="whether to plot results from previous/current runs", action="store_true")
    parser.add_argument("-g", "--gpu", type=bool, default=False, help="use GPU where possible")

    args = parser.parse_args()

    asyncio.run(main(args))
