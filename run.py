import argparse

def main():
  print("*****************************************")
  print("***** pocketML model evaluator 4000 *****")
  print("*****************************************\n")
  parser = argparse.ArgumentParser(description="Evaluation of various state of the art POS taggers, on the UD dataset")
  parser.add_argument("-v", "--verbose", help="increase output verbosity")
  parser.add_argument("model_id", type=int, help="id of the model to run")
  args = parser.parse_args()
  print(f"Running model # {args.model_id}")
  print(f"Verbose setting: {args.verbose}")

if __name__ == "__main__":
  main()
