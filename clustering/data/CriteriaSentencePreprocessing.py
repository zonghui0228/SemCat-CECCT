# *_* coding: utf-8 *_*
# @Author   : zong hui

# a series of pre-processing steps to process the criteria sentences before run MetaMap, including 
# 1. delete ordinal number
# 2. replace the ASCII code
# 3. lemmatization
# 4. delete symbols of number, operator and unit
# 5. replace the abbreviation

import os
import re
import numpy as np #1.18.5
import random
import csv
import json
import nltk #3.4
import codecs
from nltk.stem import WordNetLemmatizer #3.4

abbr_exp = "./knol/abbr.csv"
ascii_exp = "./knol/ascii.csv"

def eliminate_ordinal_number(sent):
    pattern = re.compile(r'^\(?\d+(\.|\))')
    sent = sent.strip()
    match = pattern.match(sent)
    if match:
        res = re.sub(pattern, '', sent).strip()
        return res
    else:
        return sent

def replace_ascii(sent, ascii_exp = ascii_exp):
    with codecs.open(ascii_exp, "r", encoding="utf-8") as f:
        rows = csv.reader(f)
        for row in rows:
            sent = sent.replace(row[0], row[1])
    sent = re.sub(r"\.|;$", "", sent)
    return sent

def lemmatization(sentence):
    lemmatizer = WordNetLemmatizer()
    words = nltk.word_tokenize(sentence)
    word_lower = [word.lower() for word in words]
    word_lower_pos = nltk.pos_tag(word_lower)
    word_pos = [(words[i], word_lower_pos[i][1]) for i in range(len(words))]
    words_ = []
    # print word_pos
    for word, tag in word_pos:
        if word[1:].lower() == word[1:]:
            try:
                word.encode("utf-8").decode('ascii')
                if tag.startswith('NN'):
                    word_ =  lemmatizer.lemmatize(word.lower(), pos='n')
                    if word_.lower() == word.lower(): word_ = word
                elif tag.startswith('VB'):
                    word_ = lemmatizer.lemmatize(word.lower(), pos='v')
                    if word_.lower() == word.lower(): word_ = word
                elif tag.startswith('JJ'):
                    word_ = lemmatizer.lemmatize(word.lower(), pos='a')
                    if word_.lower() == word.lower(): word_ = word
                elif tag.startswith('R'):
                    word_ = lemmatizer.lemmatize(word.lower(), pos='r')
                    if word_.lower() == word.lower(): word_ = word
                else:
                    word_ = word
            except UnicodeDecodeError:
                # print word
                word_ = ''
        else:
            try:
                word.encode("utf-8").decode('ascii')
                word_ = word
            except UnicodeDecodeError:
                word_ = ''
        words_.append(word_)
    sentence_ = ' '.join(words_)
    # print sentence_
    return sentence_

def eliminate_number_symbol_unit(sent):
    symbols1 = ">|＞|≥|≧|>=|=>|<|＜|≤|≦|≤|=<|<=|="
    symbols2 = "&lt;|&gt;"

    numeral1 = "\d+(-|~|～|\.)?\d*%?"

    unit1 = " ul| cm| mL| mg| mm| ml| kg| KG| uL"
    unit2 = "/μl|/μL|/ ul|/ uL|/㎡|/mm³|/mm^3|mm\*\*3|/mm3|/mm|/ml|/micL|/mcL|/liter|/l|/kg|/Liter|/L"
    unit3 = "U/ml|kg/m2|mL/min|IU/ml|μg/L|plts/mm3|ng/mL|mmol/liter|mmol/L|ml/min|micromole/liter|mg/dl|mg/day|mg/dL|mcg/kg|mL/minute|gm/dl|gm/L|g/l|g/dl|g/dL|g/L|g/DL|copy/ml|copies/ml|copies/mL|cells/μL|cells/uL|cells/mm3|cell/mm3|U/mL|IU/ml|IU/mL|IU/L|IU/L|G/L"

    symbol_pattern = re.compile(symbols1 + "|" + symbols2)
    numeral_pattern = re.compile(numeral1)
    unit_pattern = re.compile(unit3 + "|" + unit2 + "|" + unit1)

    sent = re.sub(symbol_pattern, "", sent)
    sent = re.sub(unit_pattern, "", sent)
    sent = re.sub(numeral_pattern, "", sent)
    # print sent
    return sent

def replace_abbreviations(sent, abbr_exp = abbr_exp):
    with open(abbr_exp, 'r') as f1:
        full_abbr = [line.strip().split(",") for line in f1]
    for [full, abbr_exp] in full_abbr:
        sent = re.sub(re.compile(abbr_exp), full, sent)
    return sent



def preprocess(infile, outfile):
    with codecs.open(infile, "r", encoding="utf-8") as inf:
        data = json.load(inf)
    n = 1    
    for criteria in data["criteria"]:
        criteria_sentence_english = criteria["criteria_sentence_english"]

        # preprocess
        en_criteria = eliminate_ordinal_number(criteria_sentence_english)
        en_criteria = replace_ascii(en_criteria)
        en_criteria = lemmatization(en_criteria)
        en_criteria = eliminate_number_symbol_unit(en_criteria)
        criteria_sentence_english_preprocessed = replace_abbreviations(en_criteria)

        criteria["criteria_sentence_english_preprocessed"] = criteria_sentence_english_preprocessed 
        print("[{}]: [before]:{}  [after]:{}".format(n, criteria_sentence_english, criteria_sentence_english_preprocessed))
        n +=1

    with codecs.open(outfile, "w", encoding="utf-8") as outf:
        json.dump(data, outf, indent=4, ensure_ascii=False)

    return "done!"

if __name__ == "__main__":
    # preprocess a criteria sentence
    text = " 3. left atrial diametrer &lt;= 55 mm;"
    text = eliminate_ordinal_number(text)
    text = replace_ascii(text)
    text = lemmatization(text)
    text = eliminate_number_symbol_unit(text)
    text_preprocessed = replace_abbreviations(text)
    print(text_preprocessed)

    # preprocess 20000 criteria sentences
    # infile = "criteria_sentences(20000).json"
    # outfile = "criteria_sentences(20000)_preprocessed.json"
    # preprocess(infile, outfile)
    print("done!")