from glob import glob
import re

# all must be run from the folder this file is in

def check_avg_sentence_length():
    sent_by_language = {}
    tok_by_language = {}
    file_paths = glob('../../data/**/ud_*/simplified/*.conllu', recursive=True)
    for file_path in file_paths:
        lang = file_path.split('\\')[-1].split('_')[0]
        print(lang)
        sent_by_language[lang] = sent_by_language.get(lang, 0)
        tok_by_language[lang] = tok_by_language.get(lang, 0)
        with open(file_path, "r", encoding="utf-8") as file_content:
            prev = ""
            for line in file_content.readlines():
                if line.strip() == "" and prev.strip() != "":
                    sent_by_language[lang] += 1
                else:
                    tok_by_language[lang] += 1
                prev = line

    print('Average sentence length by language *****')
    for (k,v) in sent_by_language.items():
        print(f'{k}: {(tok_by_language[k] / v):.2f} ({tok_by_language[k]} / {v})')

def check_eof():
    p = re.compile('[\\.\\?\\!]')
    weird = set()
    by_language = {}
    file_paths = glob('../../data/**/ud_*/simplified/*.conllu', recursive=True)
    for file_path in file_paths:
        fname = file_path.split('\\')[-1]
        by_language[fname] = 0
        with open(file_path, "r", encoding="utf-8") as file_content:
            prev = None
            for line in file_content.readlines():
                if line.strip() != "":
                    prev = line.split(None)
                    prev = prev[0]
                else:
                    if not p.match(prev) or prev != '。':
                        weird.add(f'tag: {prev}, from: {file_path}')
                        by_language[fname] += 1
    for w in weird:
        print(w)
    print(f'total weirds: {len(weird)}')
    print('by language *****')
    for k,v in by_language.items():
        print(f'{v} for {k}')

def check_num_sentences():
    by_language = {}
    file_paths = glob('../../data/**/ud_*/simplified/*.conllu', recursive=True)
    for file_path in file_paths:
        fname = file_path.split('\\')[-1]
        by_language[fname] = 0
        with open(file_path, "r", encoding="utf-8") as file_content:
            prev = ""
            for line in file_content.readlines():
                if line.strip() == "" and prev.strip() != "":
                    by_language[fname] = by_language[fname] + 1
                prev = line

    print('by language *****')
    for k,v in by_language.items():
        print(f'{k} : {v} sentences')

def check_multiword_tokens():
    by_language = {}
    file_paths = glob('../../data/**//ud_*/simplified/*.conllu', recursive=True)
    for file_path in file_paths:
        fname = file_path.split('\\')[-1]
        by_language[fname] = 0
        with open(file_path, "r", encoding="utf-8") as file_content:
            for line in file_content.readlines():
                if ' ' in line.strip():
                    by_language[fname] += 1

    print('by language *****')
    for k,v in by_language.items():
        if v > 0:
            print(f'{k} : {v} words with spaces')