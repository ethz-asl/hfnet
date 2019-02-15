#!/bin/sh

slice_num=$1

sift_ref_db="slice${slice_num}.db"
nvm_file="slice${slice_num}.nvm"
final_model="cmu-slice${slice_num}_sift_model"

# Use the provided NVM file and convert it directly to a 3D model.
mkdir ${final_model}
python3 nvm_to_model.py \
    --database_file ${sift_ref_db} \
    --nvm_file ${nvm_file} \
    --output_dir ${final_model} \

# Convert the model from a text format to a binary format.
colmap model_converter \
    --input_path ${final_model} \
    --output_path ${final_model} \
    --output_type bin
