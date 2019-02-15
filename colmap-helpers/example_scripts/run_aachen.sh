#!/bin/sh

image_dir="images_upright"
sift_db="aachen.db"
new_db="aachen-new.db"
npz_dir="aachen_npz_sfm"
nvm_file="aachen_cvpr2018_db.nvm"

match_ratio="87"
match_file="matches${match_ratio}.txt"

temp_model="aachen_temp_model"
final_model="aachen_model"

# Removove old feature and matches txt files
rm ${image_dir}/db/*.txt
rm ${match_file}

# Export txt file for the features
python3 features_from_npz.py \
    --npz_dir ${npz_dir} \
    --image_dir ${image_dir}/db/

# Match the new features using the original SIFT-based DB as a prior
python3 match_features_with_db_prior.py \
    --database_file ${sift_db} \
    --image_prefix db \
    --image_dir ${image_dir} \
    --npz_dir ${npz_dir} \
    --min_num_matches 15 \
    --num_points_per_frame 3000 \
    --use_ratio_test \
    --ratio_test_values "0.${match_ratio}"

# Create an empty Colmap DB
colmap database_creator --database_path ${new_db}

# Import the features
colmap feature_importer \
    --database_path ${new_db} \
    --image_path ${image_dir} \
    --import_path ${image_dir}

# Update the intrinsics
python3 update_db_with_nvm_intrinsics.py \
    --database_file ${new_db} \
    --nvm_file ${nvm_file}

# Import matches as two-view geometries
colmap matches_importer \
    --database_path ${new_db} \
    --match_list_path ${match_file} \
    --match_type raw \
    --SiftMatching.max_num_trials 20000 \
    --SiftMatching.min_inlier_ratio 0.20

# Build an initial model using the camera poses from the NVM file
mkdir ${temp_model}
python3 colmap_model_from_nvm.py \
    --database_file ${new_db} \
    --nvm_file ${nvm_file} \
    --output_dir ${temp_model}

# Triangulate the superpoint features using the previously prepared poses
# and the features matches stored in the DB
mkdir ${final_model}
colmap point_triangulator \
    --database_path ${new_db} \
    --image_path ${image_dir} \
    --input_path ${temp_model} \
    --output_path ${final_model}
