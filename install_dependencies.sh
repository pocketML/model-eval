conda create --name model_eval python=3.8
conda activate model_eval

# Dependencies for main evaluation program
conda install requests
conda install nltk
conda install matplotlib

# Dependencies for various models:

# Bi-LSTM Aux
pip install dynet

# SVMTool
# Run on Linux with gcc/g++ or use Windows and run on WSL (Ubuntu bash)
sudo apt-get install perl

# Bi-LSTM CRF
pip install --upgrade https://github.com/Theano/Theano/archive/master.zip
pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
pip install gensim

# Stanford POS tagger
sudo apt-get install openjdk-8-jdk

# Flair
conda install pytorch
pip install flair

# TnT
conda install dill

# CRF
conda install python-crfsuite

# Meta Tagger
conda install tensorflow