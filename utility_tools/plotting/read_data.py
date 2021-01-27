import pandas as pd
import os
from pathlib import Path
from glob import glob

CSV_PATH = os.path.join(Path(__file__).resolve().parent.parent.parent, 'results_csv')

def extract_data(metric, include_stanford):
    path = os.path.join(CSV_PATH, f"{metric}.csv")
    df = pd.read_csv(path)
    df = df.drop(df.columns[11], axis=1) # Drop Average (across language) column
    if not include_stanford:
        df = df.drop('Stanford Tagger', axis=1) # Drop Stanford column

    df = df.drop(df.index[[12]]) # Drop std dev row
    df_copy = df.drop(df.index[[10,11]]) # Drop average rows

    languages = df_copy['Language'].values.tolist()

    df = df.drop(df.columns[0], axis=1) # drop the language column

    taggers = df.columns.values.tolist()

    return df, languages, taggers

def get_average_data(metric, include_stanford=False):
    df, languages, taggers = extract_data(metric, include_stanford)
    index = 10 if include_stanford else 11
    results =  df.loc[index].tolist()

    return [languages, taggers, results]

def get_data(metric, include_stanford=False, average=False):
    df, languages, taggers = extract_data(metric, include_stanford)

    df = df.drop(df.index[[10,11]]) # Drop average rows

    # drop HMM, TnT and Brill
    #df = df.drop('Brill', axis=1).drop('HMM', axis=1).drop('TnT', axis=1)

    results = [df[x].values.flatten().tolist() for x in taggers]

    return [languages, taggers, results]
