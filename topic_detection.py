import sys
import tomotopy as tp
from pathlib import Path
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import nltk
from thefuzz import fuzz
import copy
import numpy as np
import argparse

nltk.download('stopwords', quiet=True, download_dir='/data1/multilingual_t2i/captions/venv/nltk_data')
nltk.download('punkt', quiet=True, download_dir='/data1/multilingual_t2i/captions/venv/nltk_data')

from nltk.util import ngrams
from nltk.tokenize import sent_tokenize

def make_dictionary(topic_model_file, output_file):
    bad_chars = [';', ':', '!', "*", "\"", "\'", ",", "(",")", "Labels", "Topic"]
    regex = re.compile('[_]')
    new_tokens = []
    with open(topic_model_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            for i in bad_chars:
                line = line.replace(i, '')
            tokens = line.split()
            for token in tokens:
                token = token.strip()
                if len(token) > 1 and not token.isnumeric() \
                        and (token.isalpha() or regex.search(token) \
                        and re.match(r'^-?\d+(?:\.\d+)$', token) is None):
                    token = token.replace("-","_")
                    if token not in new_tokens:
                        new_tokens.append(token)
    
    print(new_tokens)
    with open(output_file, "w") as wf:
        for tok in new_tokens:
            wf.write(tok + "\n")

def read_topic_dictionary(d_file):
    voc = []
    with open(d_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            voc.append(line.strip())
    return voc

def preprocess(input_file, topic_model_path, dictionary_file, ngram=False):
    must_words = read_topic_dictionary(d_file=dictionary_file)
    stop_words = set(stopwords.words('english'))
    concat = "_"
    bad_chars = [';', ':', '!', "*", "\"", "\'"]

    mdl = tp.LDAModel.load(topic_model_path)

    for n, line in enumerate(open(input_file, encoding='utf-8')):
        word_tokens = preprocess_instance(line, mdl, must_words, stop_words, ngram)
        infer_topic(word_tokens, mdl)

def preprocess_instance(input_document, mdl, must_words, stop_words, ngram=False):
    concat = "_"
    bad_chars = [';', ':', '!', "*", "\"", "\'"]

    for i in bad_chars:
        input_document = input_document.replace(i, '')

    input_document = input_document.replace("image","").replace("images","").replace("shows","").replace("show","")
    _line = [w for w in input_document.split() if not w.lower() in stop_words]
    line = " ".join(_line)
    word_tokens = []
    token_text = sent_tokenize(line)
    for _text in token_text:
        _text = _text.replace(".","")
        if not ngram:
            res = word_tokenize(_text.lower())
            for _res in res:
                word_tokens.append(_res)
        else:
            for i in range(1, 4):
                output = list(ngrams((_text.lower().split()),i))
                for x in output:
                    if (len(x) > 0):
                        n_text = concat.join(x)
                        if n_text not in must_words:            
                            for a_word in must_words:
                                ratio = fuzz.ratio(n_text, a_word)
                                if ratio > 90:
                                    n_text = a_word
                                    break
                        word_tokens.append(n_text)
    return word_tokens

def load_model(topic_model_path):
    return tp.LDAModel.load(topic_model_path)

def infer_topic(word_tokens, mdl):
    print(word_tokens)
    doc_inst = mdl.make_doc(word_tokens)
    topic_dist, ll = mdl.infer(doc_inst,iter=500)
    print("Topic Distribution for Unseen Docs: ", topic_dist)
    print("Log-likelihood of inference: ", ll)
        
    clone_topic = np.sort(topic_dist)[::-1]
    largest_topic = clone_topic[0]
    index_pos = topic_dist.tolist().index(largest_topic)
    topic_info = mdl.get_topic_words(index_pos, top_n=20)
    print("Target topic " +  str(largest_topic) + "\t" + str(topic_info))

def main():
    parser = argparse.ArgumentParser(description="LDA Topic Modeling CLI")
    parser.add_argument("mode", choices=["make_dictionary", "preprocess", "infer"],
                        help="Mode of operation")
    parser.add_argument("--input_file", help="Input file for preprocessing or inference")
    parser.add_argument("--model_path", default="caption_lda_100_tags.bin",
                        help="Path to the LDA model")
    parser.add_argument("--dictionary_file", default="tag_topic_dictionary.txt",
                        help="Path to the dictionary file")
    parser.add_argument("--topic_model_file", help="Topic model file for dictionary creation")
    parser.add_argument("--ngram", action="store_true", help="Use n-grams")
    parser.add_argument("--input_document", help="Input document for inference")

    args = parser.parse_args()

    if args.mode == "make_dictionary":
        if not args.topic_model_file:
            parser.error("--topic_model_file is required for make_dictionary mode")
        make_dictionary(args.topic_model_file, args.dictionary_file)
    elif args.mode == "preprocess":
        if not args.input_file:
            parser.error("--input_file is required for preprocess mode")
        preprocess(args.input_file, args.model_path, args.dictionary_file, args.ngram)
    elif args.mode == "infer":
        if not args.input_document:
            parser.error("--input_document is required for infer mode")
        must_words = read_topic_dictionary(args.dictionary_file)
        stop_words = set(stopwords.words('english'))
        mdl = load_model(args.model_path)
        word_tokens = preprocess_instance(args.input_document, mdl, must_words, stop_words, args.ngram)
        infer_topic(word_tokens, mdl)

if __name__ == "__main__":
    main()