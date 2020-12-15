import asyncio
import os
import multiprocessing
import subprocess
from datetime import datetime
from sys import argv
import platform
import argparse
import importlib
from util import data_archives
from inference import monitor_inference
from training import monitor_training, train_imported_model

# All tagger helpers will be dynamically imported at runtime.
TAGGERS = { # Entries are: model_name -> (module_name, class_name)
    "bilstm_aux": ("bilstm_aux", "BILSTMAUX"),
    "bilstm_crf": ("bilstm_crf", "BILSTMCRF"),
    "svmtool": ("svmtool", "SVMT"),
    "stanford": ("stanford", "Stanford"),
    "tnt": ("nltk_tnt", "TnT"),
    "brill": ("nltk_brill", "Brill"),
    "hmm": ("nltk_hmm", "HMM"),
    "meta_tagger": ("meta_tagger", "METATAGGER"),
    "flair": ("flair_pos", "Flair"),
    "bert_bpemb": ("bert_bpemb", "BERT_BPEMB")
}

# make sure that the name convention from the report can be translated into the one in the code
TAGGER_NAME_TRANSLATION = {
    "bilstm-plank"      : "bilstm_aux",
    "bilstm-yasunaga"   : "bilstm_crf",
    "svmtool"           : "svmtool",
    "stanford-tagger"   : "stanford",
    "tnt"               : "tnt",
    "brill"             : "brill",
    "hmm"               : "hmm",
    "meta-bilstm"       : "meta_tagger",
    "flair"             : "flair",
    "bert-bpemb"        : "bert_bpemb"
}

def import_taggers(model_names):
    """
    Dynamically import all taggers in the given list.
    This is done to avoid importing 'heavy' libraries like
    TensorFlow and PyTorch, when they are not needed.
    """
    for model_name in model_names:
        print(f"Dynamically importing tagger class '{model_name}'...")
        module_name, class_name = TAGGERS[model_name]
        module = importlib.import_module(f"taggers.{module_name}")
        TAGGERS[model_name] = module.__dict__[class_name]

def insert_arg_values(cmd, tagger, args):
    """
    Inserts values from 'args' into a command line string.
    The command string is the command line call needed to train or evaluate a given tagger.
    This string contains placeholders such as [lang] where for example the language
    being trained on will be inserted.
    """
    if (reload_str := tagger.reload_string()) is None or (args.train and not args.reload):
        reload_str = ""
    replacements = [
        ("[iters]", str(args.iter)),
        ("[model_base_path]", tagger.model_base_path()),
        ("[model_path]", tagger.model_path()),
        ("[pred_path]", tagger.predict_path()),
        ("[reload]", reload_str),
        ("[lang]", args.lang),
        ("[embeddings]", data_archives.get_embeddings_path(args.lang)),
        ("[dataset_train]", data_archives.get_dataset_path(args.lang, args.treebank, "train", simplified=tagger.simplified_dataset, eos=tagger.simplified_eos_dataset)),
        ("[dataset_test]", data_archives.get_dataset_path(args.lang, args.treebank, "test", simplified=tagger.simplified_dataset, eos=tagger.simplified_eos_dataset)),
        ("[dataset_dev]", data_archives.get_dataset_path(args.lang, args.treebank, "dev", simplified=tagger.simplified_dataset, eos=tagger.simplified_eos_dataset)),
        ("[dataset_folder]", data_archives.get_dataset_folder_path(args.lang, args.treebank, simplified=tagger.simplified_dataset, eos=tagger.simplified_eos_dataset)),
        ("[stdout]", f"[stdout_{len(tagger.predict_path())}]"),
        ("[stderr]", f"[stdout_{len(tagger.predict_path())}]")
    ]
    replaced = cmd
    for old, new in replacements:
        replaced = replaced.replace(old, new)
    return replaced.strip()

def system_call(cmd, cwd, script_location):
    """
    Perform a system call using the given command in 'cmd'.
    This is done when doing training or inference on taggers that are not
    directly imported into Python.
    """
    if platform.system() == "Windows" and cmd.startswith("bash"):
        # This is done when using the Linux Subsystem for Windows (when running SVMTool).
        split = cwd.replace(" ", "\ ").split("/")
        cwd = "/mnt/" + split[0][:-1].lower() + "/" + "/".join(split[1:])

    stderr_reroute = subprocess.STDOUT
    if "[stderr_" in cmd: # Redirect stderr to a file.
        index = cmd.index("[stderr_")
        end_stderr = index + cmd[index:].index("]")
        len_file = int(cmd[index + len("[stderr_"):end_stderr])
        file_path = cmd[end_stderr + 2: end_stderr + 2 + len_file]
        stdout_reroute = open(file_path, "w", encoding="utf-8", errors="utf-8")
        cmd = cmd[:index-1] + cmd[end_stderr + 2 + len_file:]

    stdout_reroute = subprocess.PIPE
    if "[stdout_" in cmd: # Redirect stdout to a file.
        index = cmd.index("[stdout_")
        end_stdout = index + cmd[index:].index("]")
        len_file = int(cmd[index + len("[stdout_"):end_stdout])
        file_path = cmd[end_stdout + 2: end_stdout + 2 + len_file]
        stdout_reroute = open(file_path, "w", encoding="utf-8", errors="utf-8")
        cmd = cmd[:index-1] + cmd[end_stdout + 2 + len_file:]

    script_path = f"{cwd}/{script_location}"
    if platform.system() == "Windows":
        script_path = f"\"{script_path}\"" # Add quotes around paths for safety.

    # Insert the full path to the training scripts of the tagger.
    cmd_full = cmd.replace("[script_path_train]", script_path)
    # Insert the full path to the prediction scripts of the tagger.
    cmd_full = cmd_full.replace("[script_path_test]", script_path)

    if platform.system() == "Windows" and not cmd.startswith("bash"):
        cmd_full = cmd_full.replace("/", "\\")

    print(f"Running {cmd_full}")

    if platform.system() != "Windows":
        cmd_full = cmd_full.split(" ") # This is some strange stuff wtih subprocesses.

    # Open a subprocess where the system call runs. This allows for monitoring of output.
    process = subprocess.Popen(cmd_full, stdout=stdout_reroute, stderr=stderr_reroute)
    return process

async def run_with_sys_call(args, tagger_helper, model_name, file_pointer):
    """
    This method is used when running an external tagger
    (as opposed to one being imported from f.x. NLTK).
    """
    call_train = tagger_helper.train_string() # Get training command string.
    call_infer = tagger_helper.predict_string() # Get prediction command string.

    cwd = os.getcwd().replace("\\", "/")

    final_acc = (0, 0)
    if args.train: # Train model.
        call_train = insert_arg_values(call_train, tagger_helper, args)
        process = system_call(call_train, cwd, tagger_helper.script_path_train())
        final_acc = await monitor_training(tagger_helper, process, args, model_name, file_pointer)

    model_footprint = None
    if args.eval: # Run inference task.
        if call_infer is not None:
            call_infer = insert_arg_values(call_infer, tagger_helper, args)
            process = system_call(call_infer, cwd, tagger_helper.script_path_test())
            model_footprint = await monitor_inference(tagger_helper, process)

        # Evaluate how accurate the prediction of the tagger was.
        final_acc = tagger_helper.evaluate()

    return final_acc, model_footprint

async def run_with_imported_model(args, tagger, model_name):
    """
    This method is used when running an imported tagger,
    this includes the NLTK taggers and Flair.
    """
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
        model_footprint = await monitor_inference(tagger, process)
        final_acc = pipe_1.recv() # Receive accuracy from seperate process.
    return final_acc, model_footprint

async def main(args):
    actions = "Eval" if args.eval else ("Train" if args.train else "Eval & Train")
    print(f"{actions} with models: {args.model_names} on languages: {args.langs}")

    if args.model_names == ["all"]:
        models_to_run = TAGGERS.keys() # Run training/inference on all models.
    else:
        models_to_run = []
        for name in args.model_names:
            try:
                models_to_run.append(TAGGER_NAME_TRANSLATION[name])
            except KeyError:
                print(f'{name} is not a valid tagger!')

    import_taggers(models_to_run) # Dynamically import all needed taggers.

    for model_name in models_to_run:
        if not TAGGERS[model_name].IS_IMPORTED:
            if not data_archives.archive_exists("models", model_name):
                data_archives.download_and_unpack("models", model_name)

    languages_to_use = (data_archives.LANGS_FULL.keys() - set(data_archives.LANGS_FULL.values())
                        if args.langs == ["all"] else args.langs)

    treebanks = []
    for lang in languages_to_use:
        language_full = data_archives.LANGS_FULL[lang]
        if not data_archives.archive_exists("data", language_full):
            data_archives.download_and_unpack("data", language_full)
            data_archives.transform_dataset(language_full)
        treebanks.append(data_archives.get_default_treebank(lang))

    # If neither training or inference has been specified, we do both.
    if not args.train and not args.eval:
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
        for lang, treebank in zip(languages_to_use, treebanks):
            args.treebank = treebank
            args.lang = lang
            print("=" * 60)
            print(
                f"Running '{model_name}' with '{data_archives.LANGS_FULL[lang]}' "
                f"dataset on '{args.treebank}' treebank."
            )
            file_pointer = None
            file_name = None
            if args.save_results: # Save results of runs to a file.
                if not os.path.exists("results"):
                    os.mkdir("results")
                formatted_date = datetime.now().strftime("%Y-%m-%d_%H.%M")
                model_dir = f"results/{model_name}"
                if not os.path.exists(model_dir):
                    print(f"Creating result directory for model: {model_dir}")
                    os.mkdir(model_dir)
                language_dir = f"{model_dir}/{lang}_{treebank}"
                if not os.path.exists(language_dir):
                    print(f"Creating result directory for language: {model_dir}")
                    os.mkdir(language_dir)
                file_name = f"{language_dir}/{formatted_date}.out"
                file_pointer = open(file_name, "w")

            # Load model immediately if we are only evaluating, or if we are continuing training.
            load_model = (args.eval and not args.train) or (args.reload and args.train)
            tagger = TAGGERS[model_name](args, model_name, load_model) # Create tagger helper class.

            print(f"Tagger code size: {tagger.code_size() // 1000} KB")
            if load_model:
                print(f"Tagger model size: {tagger.model_size() // 1000} KB")

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
                memory_usage, code_size, model_size, size_compressed = model_footprint
                print(
                    f"Model footprint - Memory: {memory_usage} KB | "+
                    f"Code: {code_size} KB | Model: {model_size} KB | " +
                    f"Compressed: {size_compressed} KB"
                )

            if file_pointer is not None: # Save final test-set/prediction accuracy.
                file_pointer.write(f"Final token acc: {token_acc}\n")
                file_pointer.write(f"Final sentence acc: {sent_acc}\n")

                if model_footprint is not None: # Save size of model footprint.
                    memory_usage, code_size, model_size, size_compressed = model_footprint
                    file_pointer.write(f"Memory usage: {memory_usage}\n")
                    file_pointer.write(f"Code size: {code_size}\n")
                    file_pointer.write(f"Model size: {model_size}\n")
                    file_pointer.write(f"Compressed size: {size_compressed}")

                print(f"Wrote results of run to '{file_name}'")
                file_pointer.close()

def print_logo():
    logo_str = (
        "***********************************************************************************\n"
        "***********************************************************************************\n"
        "***                                                                             ***\n"
        "***                             █|                 █|      █|      █| █|        ***\n"
        "***  █████|     ███|     █████| █|  █|    ███|   ███████|  ███|  ███| █|        ***\n"
        "***  █|    █| █|    █| █|       ███|    ███|███|   █|      █|  █|  █| █|        ***\n"
        "***  █|    █| █|    █| █|       █|  █|  █|         █|      █|      █| █|        ***\n"
        "***  █████|     ███|     █████| █|    █|  █████|     ███|  █|      █| ███████|  ***\n"
        "***  █|                                                                         ***\n"
        "***  █|          ★·.·`¯´·._pos_tagger_evaluation_tool_.·´¯`·.·★                ***\n"
        "***                                                                             ***\n"
        "***********************************************************************************\n"
        "***********************************************************************************\n"
    )
    print(logo_str)

if __name__ == "__main__":
    print_logo()

    parser = argparse.ArgumentParser(description="Evaluation of various POS taggers on UD datasets")

    # Required arguments (positionals)
    choices_models = list(TAGGER_NAME_TRANSLATION.keys()) + ["all"]
    parser.add_argument("model_names", type=str, choices=choices_models, nargs="+", help="name of the model to run")

    # Optional arguments
    choices_langs = list(data_archives.LANGS_FULL.keys()) + ["all"]
    parser.add_argument("-tg", "--tag", nargs="+", default=[])
    parser.add_argument("-l", "--langs", type=str, choices=choices_langs, nargs="+", default=["en"], help="choose dataset language(s). Default is English.")
    parser.add_argument("-i", "--iter", type=int, default=10, help="number of training iterations. Default is 10.")
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    parser.add_argument("-tb", "--treebank", type=str, help="UD treebank to use as dataset (fx. 'gum')", default=None, required=False)
    parser.add_argument("-r", "--reload", help="whether to load a saved model for further training", action="store_true")
    parser.add_argument("-lb", "--loadbar", help="whether to run with loadbar", action="store_true")
    parser.add_argument("-s", "--save-results", help="whether to save accuracy & size complexity measurements", action="store_true")
    parser.add_argument("-t", "--train", help="whether to train the given model", action="store_true")
    parser.add_argument("-e", "--eval", help="whether to predict & evaluate accuracy using the given model", action="store_true")
    parser.add_argument("-m", "--max-iter", help="where to stop when 'iter' iterations has been run during training", action="store_true")
    parser.add_argument("-g", "--gpu", help="use GPU where possible", action="store_true")
    parser.add_argument("-c", "--config", type=str, help="path to a config file from which to read in arguments")

    args_from_file = None
    if len(argv) == 3 and argv[1] in ("-c", "--config"):
        # Load command line arguments from a config file.
        with open(argv[2]) as fp:
            args_from_file = fp.readline().split(None)

    args = parser.parse_args(args_from_file)

    iso_langs = []
    if args.langs == ["all"]:
        iso_langs = list(set(data_archives.LANGS_ISO.values()))
    else:
        for lang in args.langs:
            iso_langs.append(data_archives.LANGS_ISO[lang])

    args.langs = iso_langs

    asyncio.run(main(args))
