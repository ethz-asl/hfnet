#!/bin/sh

image_dir="images_upright"
sift_db="aachen.db"
nvm_file="aachen_cvpr2018_db.nvm"

temp_model="aachen_temp_sift_model"
final_model="aachen_sift_model"

# Update the provided SIFT DB file with the correct intrinsics.
python3 update_db_with_nvm_intrinsics.py \
    --database_file ${sift_db} \
    --nvm_file ${nvm_file}

# Use the provided NVM file for ground-truth poses of all the cameras.
mkdir ${temp_model}
python3 colmap_model_from_nvm.py \
    --database_file ${sift_db} \
    --nvm_file ${nvm_file} \
    --output_dir ${temp_model}

# Triangulate the 3D model according to the matches provided in the SIFT DB.
mkdir ${final_model}
colmap point_triangulator \
    --database_path ${sift_db} \
    --image_path ${image_dir} \
    --input_path ${temp_model} \
    --output_path ${final_model}
