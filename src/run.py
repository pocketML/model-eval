import argparse
import data_archives
import transform_data

def main():
    print("*****************************************")
    print("***** pocketML model evaluator 4000 *****")
    print("*****************************************\n")
    parser = argparse.ArgumentParser(description="Evaluation of various state of the art POS taggers, on the UD dataset")
    parser.add_argument("-v", "--verbose", help="increase output verbosity")
    parser.add_argument("model_name", type=str, help="name of the model to run")
    args = parser.parse_args()
    print(f"Running model # {args.model_name}")
    print(f"Verbose setting: {args.verbose}")

    if not data_archives.archive_exists("data"):
        data_archives.download_and_unpack("data")
        transform_data.transform_datasets()
    if not data_archives.archive_exists("models"):
        data_archives.download_and_unpack("models")

if __name__ == "__main__":
    main()
