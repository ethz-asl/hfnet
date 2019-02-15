#!/bin/sh

slice_num=$1

image_dir="images"
sift_ref_db="slice${slice_num}.db"
sift_query_db="query_slice${slice_num}.db"
new_db="cmu-slice${slice_num}_new.db"
npz_dir="cmu_npz_sfm"
nvm_file="slice${slice_num}.nvm"

sift_feature_dir="sift/sift_descriptors"
query_txt_file="slice${slice_num}.queries_with_intrinsics.txt"

match_ratio="85"
match_file="matches${match_ratio}.txt"

temp_model="cmu-slice${slice_num}_temp_model"
final_model="cmu-slice${slice_num}_model"


# Creates a tentative reference DB for SIFT as we need prior matches later on.
python3 magic_cmu_to_db.py \
    --sift_feature_dir ${sift_feature_dir} \
    --nvm_file ${nvm_file}\
    --database_file ${sift_ref_db}

colmap exhaustive_matcher --database_path ${sift_ref_db}

# Create a query SIFT DB for our localization evaluation algorithms.
python3 create_cmu_query_db.py
    --sift_feature_dir ${sift_feature_dir} \
    --query_txt_file ${query_txt_file}\
    --database_file ${sift_query_db}

# Match the new features using the original SIFT-based DB as a prior
python3 features_from_npz.py \
    --npz_dir ${spz_dir}/slice${slice_num}/database \
    --image_dir ${image_dir}/database/

python3 match_features_with_db_prior.py \
    --database_file ${sift_ref_db} \
    --image_prefix "" \
    --image_dir ${image_dir} \
    --npz_dir ${spz_dir}/slice${slice_num}/ \
    --min_num_matches 15 \
    --num_points_per_frame 3000 \
    --use_ratio_test \
    --ratio_test_values "0.${match_ratio}"

# Create an empty Colmap DB
colmap database_creator --database_path ${new_db}

# Import the features
colmap feature_importer \
    --database_path ${new_db} \
    --image_path${image_dir} \
    --import_path ${image_dir}

# Update the intrinsics using the ones stored in the NVM file.
python3 update_db_with_nvm_intrinsics.py \
    --database_file ${new_db} \
    --nvm_file ${nvm_file}

# Necessary as no principal points are stored in the NVM file.
python3 update_db_cmu_with_intrinsics.py --database_file ${new_db}

# Import matches as two-view geometries
colmap matches_importer \
    --database_path ${new_db} \
    --match_list_path ${match_file} \
    --match_type raw

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
    --input_path ${temp_model}  \
    --output_path ${final_model} \
