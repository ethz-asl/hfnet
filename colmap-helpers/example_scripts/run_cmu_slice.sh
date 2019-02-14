#!/bin/sh

slice_num=$1
slice_db=slice${slice_num}.db

# Creates a tentative SIFT-based db as we need prior matches for SuperPoint matching later on.
python3 ../colmap-helpers/magic_cmu_to_db.py --slice_num $1
colmap exhaustive_matcher --database_path ${slice_db}

# Query SIFT DB for our localization evaluation algorithms.
python3 ../colmap-helpers/get_query_db.py --slice_num $1

# Import the precomputed features and match them using the prior from the original SIFT db file.
python3 ../colmap-helpers/features_from_npz.py --npz_dir cmu-suburban_resize-1024_sfm-nms4/slice${slice_num}/database --image_dir images/database/
ratio=85
python3 ../colmap-helpers/match_features_with_db_prior.py --database_file ${slice_db} --image_prefix "" --image_dir "images" --npz_dir cmu-suburban_resize-1024_sfm-nms4/slice${slice_num}/ --min_num_matches=15 --num_points_per_frame=3000 --use_ratio_test --ratio_test_values "0.${ratio}"

sp_db=slice${slice_num}_${ratio}.db
colmap database_creator --database_path ${sp_db}
colmap feature_importer --database_path ${sp_db} --image_path images/ --import_path images/
# Update the intrinsics using the ones stored in the NVM file.
python3 ../colmap-helpers/update_db_with_nvm_intrinsics.py --database_file ${sp_db} --nvm_file slice${slice_num}.nvm
# Necessary as no principal points are stored in the NVM file.
python3 ../colmap-helpers/update_db_cmu_with_intrinsics.py --database_file ${sp_db}
colmap matches_importer --database_path ${sp_db} --match_list_path matches${ratio}.txt --match_type raw

# Import the ground-truth camera poses from the NVM file and build the initial model structure.
model_dir=model_slice${slice_num}_${ratio}
mkdir ${model_dir}
python3 ../colmap-helpers/colmap_model_from_nvm.py --database_file ${sp_db} --nvm_file slice${slice_num}.nvm --output_dir ${model_dir}

# Use the matches in the DB file to triangulate the 3D features.
triangulated_dir=triangulated_slice${slice_num}_${ratio}
mkdir ${triangulated_dir}
colmap point_triangulator --database_path ${sp_db} \
  --image_path images/ \
  --input_path ${model_dir}  \
  --output_path ${triangulated_dir} \
