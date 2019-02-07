#!/bin/sh
#echo 'Removing old feature txt files'
#rm images/overcast-reference/left/*.txt
#rm images/overcast-reference/right/*.txt
#rm images/overcast-reference/rear/*.txt

#echo 'Doing the real work'
#python colmap-helpers/features_from_npz.py --npz_dir robotcar_resize-960_sfm-nms4/overcast-reference/left/ --image_dir images/overcast-reference/left/
#python colmap-helpers/features_from_npz.py --npz_dir robotcar_resize-960_sfm-nms4/overcast-reference/right/ --image_dir images/overcast-reference/right/
#python colmap-helpers/features_from_npz.py --npz_dir robotcar_resize-960_sfm-nms4/overcast-reference/rear/ --image_dir images/overcast-reference/rear/

#rm matches*.txt

#python colmap-helpers/match_features_with_db_prior.py --database_file overcast-reference.db --image_prefix "" --image_dir images/ --npz_dir robotcar_resize-960_sfm-nms4 --min_num_matches=15 --num_points_per_frame=3000 --use_ratio_test --ratio_test_values "0.92"

#rm robotcar92.db
#colmap database_creator --database_path robotcar92.db
#colmap feature_importer --database_path robotcar92.db --image_path images/ --import_path images/
#python colmap-helpers/update_db_with_nvm_intrinsics.py --database_file robotcar92.db --nvm_file all.nvm
#python colmap-helpers/update_db_intrinsics_from_another_db.py --intrinsics_database_file overcast-reference.db --database_file_to_modify robotcar92.db
#colmap matches_importer --database_path robotcar92.db --match_list_path matches92.txt --match_type raw

mkdir model_robotcar92_nms4
python colmap-helpers/colmap_model_from_nvm.py --database_file robotcar92.db --nvm_file all.nvm --output_dir model_robotcar92_nms4/

mkdir triangulated_model_robotcar92_nms4
colmap point_triangulator --database_path robotcar92.db \
  --image_path images/ \
  --input_path model_robotcar92_nms4  \
  --output_path triangulated_model_robotcar92_nms4
