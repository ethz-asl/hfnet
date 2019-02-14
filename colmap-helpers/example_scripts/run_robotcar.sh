#!/bin/sh

rm images/overcast-reference/left/*.txt
rm images/overcast-reference/right/*.txt
rm images/overcast-reference/rear/*.txt

# Import precomputed features.
python3 colmap-helpers/features_from_npz.py --npz_dir robotcar_resize-960_sfm-nms4/overcast-reference/left/ --image_dir images/overcast-reference/left/
python3 colmap-helpers/features_from_npz.py --npz_dir robotcar_resize-960_sfm-nms4/overcast-reference/right/ --image_dir images/overcast-reference/right/
python3 colmap-helpers/features_from_npz.py --npz_dir robotcar_resize-960_sfm-nms4/overcast-reference/rear/ --image_dir images/overcast-reference/rear/

rm matches*.txt

# Use the original SIFT-based DB as a prior and match the superpoint features.
python3 colmap-helpers/match_features_with_db_prior.py --database_file overcast-reference.db --image_prefix "" --image_dir images/ --npz_dir robotcar_resize-960_sfm-nms4 --min_num_matches=15 --num_points_per_frame=3000 --use_ratio_test --ratio_test_values "0.85"

# Create a COLMAP DB, import features, update intrinsics and import matches as two-view geometries.
rm robotcar85.db
colmap database_creator --database_path robotcar85.db
colmap feature_importer --database_path robotcar85.db --image_path images/ --import_path images/
python3 colmap-helpers/update_db_with_nvm_intrinsics.py --database_file robotcar85.db --nvm_file all.nvm
python3 colmap-helpers/update_db_intrinsics_from_another_db.py --intrinsics_database_file overcast-reference.db --database_file_to_modify robotcar85.db
colmap matches_importer --database_path robotcar85.db --match_list_path matches85.txt --match_type raw

# Build an initial model using the camera poses from the NVM file.
mkdir model_robotcar85_nms4
python3 colmap-helpers/colmap_model_from_nvm.py --database_file robotcar85.db --nvm_file all.nvm --output_dir model_robotcar85_nms4/

# Triangulate the superpoint features using the previously prepared poses and the features matches stored in DB. COLMAP model will be the output.
mkdir triangulated_model_robotcar85_nms4
colmap point_triangulator --database_path robotcar85.db \
  --image_path images/ \
  --input_path model_robotcar85_nms4  \
  --output_path triangulated_model_robotcar85_nms4
