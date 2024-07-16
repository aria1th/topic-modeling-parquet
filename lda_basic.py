import sys
import os
import tomotopy as tp
from pathlib import Path
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.util import ngrams
import argparse
import nltk

nltk.download('stopwords', download_dir='/data1/multilingual_t2i/captions/venv/nltk_data')
nltk.download('punkt', download_dir='/data1/multilingual_t2i/captions/venv/nltk_data')

def lda_example(input_file, save_path, output_file, use_tags=False, ngram=False):
    stop_words = set(stopwords.words('english'))
    concat = "_"
    bad_chars = [';', ':', '!', "*", "\"", "\'"]
    my_file = Path(save_path)
    if not my_file.exists():
        mdl = tp.LDAModel(tw=tp.TermWeight.ONE, min_cf=3, rm_top=10, k=100)
        for n, line in enumerate(open(input_file, encoding='utf-8')):
            for i in bad_chars:
                line = line.replace(i, '')

            if use_tags:
                word_tokens = [x.strip().replace(" ", "_") for x in line.split(',')]
            else:
                _line = [w for w in line.split() if not w.lower() in stop_words]
                line = " ".join(_line)
                word_tokens = []
                token_text = sent_tokenize(line)
                for _text in token_text:
                    _text = _text.replace(".", "")
                    if not ngram:
                        res = word_tokenize(_text.lower())
                        word_tokens.extend(res)
                    else:
                        for i in range(2, 3):
                            output = list(ngrams((_text.lower().split()), i))
                            for x in output:
                                if len(x) > 0:
                                    n_text = concat.join(x)
                                    word_tokens.append(n_text)

            mdl.add_doc(word_tokens)

        mdl.burn_in = 100
        mdl.train(0)
        print('Num docs:', len(mdl.docs), ', Vocab size:', len(mdl.used_vocabs), ', Num words:', mdl.num_words)
        print('Removed top words:', mdl.removed_top_words)
        print('Training...', file=sys.stderr, flush=True)
        mdl.train(1000, show_progress=True)
        mdl.summary()
        print('Saving...', file=sys.stderr, flush=True)
        mdl.save(save_path, True)
    else:
        print("Loading the model...")
        mdl = tp.LDAModel.load(save_path)

    extractor = tp.label.PMIExtractor(min_cf=10, min_df=5, max_len=5, max_cand=10000, normalized=True)
    cands = extractor.extract(mdl)
    labeler = tp.label.FoRelevance(mdl, cands, min_df=5, smoothing=1e-2, mu=0.25)
    with open(output_file, "w") as f:
        for k in range(mdl.k):
            t = f"== Topic #{k} =="
            f.write(t + "\n")
            print(t)
            labels = ', '.join(label for label, score in labeler.get_topic_labels(k, top_n=10))

            print("Labels:", labels)
            f.write("Labels: " + labels + "\n")
            for word, prob in mdl.get_topic_words(k, top_n=15):
                print(word, prob, sep='\t')
                f.write(f"{word}\t{prob}\t")
            print()
            f.write("\n")

def main():
    parser = argparse.ArgumentParser(description="Run LDA topic modeling on input file")
    parser.add_argument("--input_file", help="Path to the input file", required=True)
    parser.add_argument("--save_path", help="Path to save/load the LDA model", required=True)
    parser.add_argument("--output_file", help="Path to save the output file", default=None)
    parser.add_argument("--tag", action="store_true", help="Use tag-based tokenization")
    parser.add_argument("--ngram", action="store_true", help="Use n-grams (only for non-tag mode)")

    args = parser.parse_args()

    if args.tag:
        if not args.output_file:
            print("Warning: No output file specified. Using default output file.")
            args.output_file = "./runs/topic_model_tags_100.txt"
            os.makedirs("./runs", exist_ok=True)
        print('Running LDA for tags')
    else:
        if not args.output_file:
            print("Warning: No output file specified. Using default output file.")
            args.output_file = "./runs/topic_model_natural_100.txt"
            os.makedirs("./runs", exist_ok=True)
        print('Running LDA for captions')

    lda_example(args.input_file, args.save_path, args.output_file, use_tags=args.tag, ngram=args.ngram)

if __name__ == "__main__":
    main()