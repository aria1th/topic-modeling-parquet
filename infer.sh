#!/bin/bash

python=./venv/bin/python
data_type="TAG" # or "NL"
file_to_analyze=./captions/danbooru2023.parquet
run_path=./captions/runs/danbooru2023parquet
use_ngram=0 # or 1

cd ./captions/cli
# Create run directory if it doesn't exist
mkdir -p $run_path

# Set variables based on data_type
if [ "$data_type" = "TAG" ]; then
    column_name="parsed"
    tag_option="--tag"
else
    column_name="parsed"
    tag_option=""
fi

# Set ngram option
if [ $use_ngram -eq 1 ]; then
    ngram_option="--ngram"
else
    ngram_option=""
fi

# Convert parquet to csv
$python convert_parquet_to_csv.py \
    --input_file $file_to_analyze \
    --output_file $run_path/captions.csv \
    --column $column_name

echo "Data conversion completed. Starting LDA..."
# Run LDA
$python lda_basic.py \
    --input_file $run_path/captions.csv \
    --save_path $run_path/lda_model.bin \
    --output_file $run_path/topic_model.txt \
    $tag_option $ngram_option

echo "LDA process completed. Starting inference..."
# Create dictionary
$python topic_detection.py make_dictionary \
    --topic_model_file $run_path/topic_model.txt \
    --dictionary_file $run_path/topic_dictionary.txt

echo "Dictionary creation completed."
# Example inference (you can modify this part as needed)
$python topic_detection.py infer \
    --model_path $run_path/lda_model.bin \
    --dictionary_file $run_path/topic_dictionary.txt \
    --input_document "Example document for inference" \
    $ngram_option

echo "Example inference completed."
echo "Inference process completed. Results are saved in $run_path"