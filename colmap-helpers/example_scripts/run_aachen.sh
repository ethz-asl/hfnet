#!/bin/sh
echo 'Removing old feature txt files'
rm images_upright/db/*.txt

echo 'Unpacking the features and preparing the model'
python3 colmap-helpers/features_from_npz.py --npz_dir db_nms4 --image_dir images_upright/db/

rm matches*.txt

# Use the original SIFT-based DB as a prior and match the superpoint features.
python3 colmap-helpers/match_features_with_db_prior.py --database_file aachen.db --image_prefix db --image_dir images_upright --npz_dir db_nms4 --min_num_matches=15 --num_points_per_frame=3000 --use_ratio_test --ratio_test_values "0.87"

# Create a COLMAP DB, import features, update intrinsics and import matches as two-view geometries.
colmap database_creator --database_path database87.db
colmap feature_importer --database_path database87.db --image_path images_upright/ --import_path images_upright/
python3 colmap-helpers/update_db_with_nvm_intrinsics.py --database_file database87.db --nvm_file aachen_cvpr2018_db.nvm
colmap matches_importer --database_path database87.db --match_list_path matches87.txt --match_type raw --SiftMatching.max_num_trials 20000 --SiftMatching.min_inlier_ratio 0.20

# Build an initial model using the camera poses from the NVM file.
mkdir initial_model
python3 colmap-helpers/colmap_model_from_nvm.py --database_file database87.db --nvm_file aachen_cvpr2018_db.nvm --output_dir initial_model/

# Triangulate the superpoint features using the previously prepared poses and the features matches stored in DB. COLMAP model will be the output.
mkdir triangulated_model
colmap point_triangulator --database_path database87.db --image_path images_upright/ --input_path initial_model     --output_path triangulated_model
