@ECHO OFF

conda create --name model_eval python=3.8
conda activate model_eval

Rem Dependencies for main evaluation program
conda install requests
conda install nltk
conda install matplotlib

Rem Dependencies for various models:

Rem Bi-LSTM Aux
pip install dynet

Rem SVMTool
Rem Run on Linux with gcc/g++ or use Windows and run on WSL (Ubuntu bash)
choco install strawberryperl

Rem Bi-LSTM CRF
pip install --upgrade https://github.com/Theano/Theano/archive/master.zip
pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
pip install gensim

Rem Stanford POS tagger
choco install jdk8

Rem Flair
conda install pytorch
pip install flair

Rem TnT
conda install dill

Rem CRF
conda install python-crfsuite

Rem Meta Tagger
conda install tensorflow