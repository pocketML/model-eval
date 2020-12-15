# pocketML
Training and evaluating classical and modern POS taggers on Universal Dependencies datasets.

## Usage
The core of the program is accessed through `src/run.py`. This file accepts the name of one or more 
POS taggers as a required argument. See below for a list of valid taggers.

* bilstm-plank
* bilstm-yasanuga
* svmtool
* stanford-tagger
* tnt
* brill
* hmm
* meta-tagger
* flair
* bert-bpemb

See the file `installation_guide.txt` for a list of dependencies needed to run these taggers.

Optional arguments are outlined below.
* `-l` - Specify one or more languages to use, either as language code or English name
* `-i` - How many iterations to train tagger(s) for 
* `-t` - Flag - Train the given tagger(s)
* `-e` - Flag - Evaluate prediction accuracy of tagger(s)
* `-m` - Flag - Force stop training after -i iters
* `-r` - Flag - Reload and continue training tagger(s)
* `-v` - Flag - Provide verbose output
* `-s` - Flag - Save results of evaluation to a file
* `-c` - Path to config file to load cmd args from

## Examples
### Train the BiLSTM model by Plank et al. for 50 maximum iterations on the english dataset with verbose output

`python src/run.py bilstm-plank -t -i 50 -l -m english -v`

### Evaluate the Stanford Tagger on the spanish dataset and save results to a file

`python src/run.py stanford -e -l spanish -s`

### Load command line arguments from a file called `config.txt`

`python src/run.py -c config.txt`
