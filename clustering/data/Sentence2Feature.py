# coding:utf-8
# @Author   : zong hui

# parse the metamap results, and generate UMLS-semantic types based feature matrix by term frequency

import re
import os
import csv
import json
import codecs

def _get_umls_semantic_types(infile):
    # 127 UMLS Semantic Type
    umls_semantic_type, umls_semantic_type_abbr = [], []
    with open(infile, "r") as f:
        for line in f:
            l = line.strip().split("|")
            umls_semantic_type.append(l[2])
            umls_semantic_type_abbr.append(l[0])
    return umls_semantic_type, umls_semantic_type_abbr
# umls_semantic_type, umls_semantic_type_abbr = _get_umls_semantic_types(umls_semantic_types_file)

def _parse_metamap_results(text):
    terms = []  # all terms
    lines = text.split("\n")
    for line in lines:
        # print line
        line = line.strip()
        if line.startswith("Phrase"):
            Phrase = line.strip()
        else:
            CUI = re.compile(r"\d+   C\d+:").findall(line)
            if CUI:
                MetaMap_CUI = CUI[0][CUI[0].index("C"):-1]
                # Semantic type
                if re.compile(r"\[([^\[]+)\]$").findall(line):
                    Semantic_Type = re.compile(r"\[([^\[]+)\]$").findall(line)[0]
                else:
                    Semantic_Type = ""
                # MetaMap Word
                if re.compile(r" \(.+\) \[").findall(line):
                    MetaMap_Word = re.compile(r" \(.+\) \[").findall(line)[0][2:-3]
                else:
                    MetaMap_Word = ""
                # raw word
                # Raw_Word = re.compile(r":.+\[").findall(line)[0][1:-1].split(" (")[0]
                Raw_Word = line.replace("["+Semantic_Type+"]", "").replace(" ("+MetaMap_Word+") ", "")
                Raw_Word = re.sub(r"\d+   C\d+:", "", Raw_Word)
                Semantic_Type_mul = [s.replace("@", ", ") for s in Semantic_Type.replace(", ", "@").split(",")]
                for Sem_type in Semantic_Type_mul:
                    term = [MetaMap_CUI, Sem_type, MetaMap_Word, Raw_Word, Phrase]
                    terms.append(term)
            else:
                pass
    semantic_types = [t[1] for t in terms]
    return semantic_types
# text = """Processing 00000000.tx.1: combine extensor tendon injury\n\nPhrase: combine\nMeta Mapping (1000):\n  1000   C0336789:Combine [Manufactured Object]\n\nPhrase: extensor tendon injury\nMeta Mapping (901):\n   660   C1184148:Extensor [Body Location or Region]\n   901   C0039504:TENDON INJURY (Tendon Injuries) [Injury or Poisoning]\nMeta Mapping (901):\n   734   C0224849:Extensor tendon (Structure of extensor tendon) [Body Part, Organ, or Organ Component]\n   827   C3263722:Injury (Traumatic AND/OR non-traumatic injury) [Injury or Poisoning]\n"""
# _parse_metamap_results(text)

def _get_feature_matrix(infile, outfile):
    umls_semantic_type, umls_semantic_type_abbr = _get_umls_semantic_types(umls_semantic_types_file)
    print(umls_semantic_type_abbr)

    with codecs.open(infile, "r", encoding="utf-8") as inf:
        data = json.load(inf)

    feature_matrix = []
    for criteria in data["criteria"]:
        count = criteria["No."]
        print(count)
        metamap_result = criteria["MetaMap_results"]
        semantic_types = _parse_metamap_results(metamap_result)
        feature_vector = [0 for i in range(len(umls_semantic_type))]
        for s in semantic_types:
            value = semantic_types.count(s) / len(semantic_types)
            ix = umls_semantic_type.index(s)
            feature_vector[ix] = value
        feature_matrix.append([count] + feature_vector)
    with open(outfile, "w", newline='') as outf:
        csv_write = csv.writer(outf)
        csv_write.writerow([]+umls_semantic_type_abbr)
        csv_write.writerows(feature_matrix)
 
if __name__ == "__main__":
    umls_semantic_types_file = "./knol/SemanticTypes_2018AB.txt"
    metamp_results_file = "criteria_sentences_preprocessed_metamap_filter(19185).json"
    feature_matrix_file = "feature_matrix_data.csv"       
    _get_feature_matrix(metamp_results_file, feature_matrix_file)
    print("done!")
