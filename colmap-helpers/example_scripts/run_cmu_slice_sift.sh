#!/bin/sh

slice_num=$1

slice_db="slice${slice_num}.db"
final_model="cmu-slice${slice_num}_sift_model"

mkdir ${final_model}

# Use the provided NVM file and convert it directly to a 3D model.
python3 nvm_to_model.py \
    --slice ${slice_num} \
    --output_dir ${final_model}

# Convert the model from a text format to a binary format.
colmap model_converter \
    --input_path ${final_model} \
    --output_path ${final_model} \
    --output_type bin

