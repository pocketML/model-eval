import os
import argparse
import data_archives
import transform_data

MODELS_SYS_CALLS = {
    "bilstm": (
        "python3 [dir]/models/bilstm-aux/src/structbilty.py --dynet-mem 1500 " +
        "--train data/UD_English-GUM/en_gum-ud-train.conllu " +
        "--test data/UD_English-GUM/en_gum-ud-test.conllu --iters 10 --model en"
    ),
    "svmtool": "perl [dir]/models/svmtool/bin/SVMTlearn.pl -V 2 models/svmtool/bin/config.svmt"
}

def system_call(cmd):
    curr_dir = "/" + os.getcwd().replace("\\", "/")
    cmd_full = cmd.replace("[dir]", curr_dir)
    print(f"Running {cmd_full}")
    os.system(f"bash -c \"{cmd_full}\"")

def main():
    print("*****************************************")
    print("***** pocketML model evaluator 4000 *****")
    print("*****************************************\n")
    
    parser = argparse.ArgumentParser(description="Evaluation of various state of the art POS taggers, on the UD dataset")
    
    # required arguments (positionals)
    parser.add_argument("model_name", type=str, choices=MODELS_SYS_CALLS.keys(), help="name of the model to run")

    # optional arguments
    parser.add_argument("-l", "--lang", type=str, default="en", help="choose dataset language. Default is English.")
    parser.add_argument("-i", "--iter", type=int, default=10, help="number of training iterations. Default is 10.")
    parser.add_argument("-v", "--verbose", help="increase output verbosity")
    
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

    system_call(sys_call)

if __name__ == "__main__":
    main()
