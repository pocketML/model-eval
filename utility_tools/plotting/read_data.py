import pandas as pd
import os
from pathlib import Path
from glob import glob

CSV_PATH = os.path.join(os.path.join(Path(__file__).resolve().parent.parent.parent, 'results'), 'csv')

def get_dataframe(path):
    return pd.read_csv(path)

def get_accuracy_data(token=True):
    if token:
        path = os.path.join(CSV_PATH, 'results_token.csv')
    else:
        path = os.path.join(CSV_PATH, 'results_sentence.csv')
    df = get_dataframe(path)
    df = df.drop(df.columns[11], axis=1).drop('Stanford Tagger', axis=1) # drop Stanford and Average column
    df = df.drop(df.index[[10,11,12]]) # Drop average 4 lang, all lang, and std dev rows
    languages = df['Language'].values.tolist()
    df = df.drop(df.columns[0], axis=1) # drop the language column
    # drop HMM, TnT and Brill
    #df = df.drop('Brill', axis=1).drop('HMM', axis=1).drop('TnT', axis=1)
    taggers = df.columns.values.tolist()
    results = [df[x].values.flatten().tolist() for x in taggers]
    return [languages, taggers, results]

def get_size_data():
    path = path = os.path.join(CSV_PATH, 'results_sizes.csv')
    df = get_dataframe(path)
    taggers = df['Tagger'].values.tolist()
    df = df.drop(df.columns[0], axis=1)
    metrics = df.columns.values.tolist()
    results = [df.iloc[i].values.tolist() for i in range(len(taggers))]
    return [metrics, taggers, results]