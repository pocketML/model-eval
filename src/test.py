from time import sleep
from os import system

cmd = (
    "python models/bilstm-aux/src/structbilty.py --dynet-mem 1500 " +
    "--train data/UD_English-GUM/simplified/en_gum-ud-train.conllu " +
    "--test data/UD_English-GUM/simplified/en_gum-ud-test.conllu --iters 10 --model en"
)

system(cmd)
