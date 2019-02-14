#!/bin/sh

slice_num=$1
slice_db=slice${slice_num}.db

model_dir=model_sift_slice${slice_num}
triangulated_dir=triangulated_sift_nvm_slice${slice_num}

rm -rf ${triangulated_dir}

mkdir ${triangulated_dir}

# Use the provided NVM file and convert it directly to a 3D model.
python3 colmap-helpers/nvm_to_model.py --slice ${slice_num} --output_dir ${triangulated_dir}
# Convert the model from a text format to a binary format.
colmap model_converter --input_path ${triangulated_dir} --output_path ${triangulated_dir} --output_type bin
rm ${triangulated_dir}/*.txt
