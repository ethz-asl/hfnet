#!/bin/sh

slice_num=$1
slice_db=slice${slice_num}.db

python colmap-helpers/magic_cmu_to_db.py --slice_num $1
python colmap-helpers/get_query_db.py --slice_num $1
colmap exhaustive_matcher --database_path ${slice_db}

python colmap-helpers/features_from_npz.py --npz_dir cmu-suburban_resize-1024_sfm-nms4/slice${slice_num}/database --image_dir images/database/

ratio=85
python colmap-helpers/match_features_with_db_prior.py --database_file ${slice_db} --image_prefix "" --image_dir "images" --npz_dir cmu-suburban_resize-1024_sfm-nms4/slice${slice_num}/ --min_num_matches=15 --num_points_per_frame=3000 --use_ratio_test --ratio_test_values "0.${ratio}"

sp_db=slice${slice_num}_${ratio}.db
colmap database_creator --database_path ${sp_db}
colmap feature_importer --database_path ${sp_db} --image_path images/ --import_path images/
python colmap-helpers/update_db_with_nvm_intrinsics.py --database_file ${sp_db} --nvm_file slice${slice_num}.nvm
colmap matches_importer --database_path ${sp_db} --match_list_path matches${ratio}.txt --match_type raw

model_dir=model_slice${slice_num}_${ratio}
mkdir ${model_dir}
python colmap-helpers/colmap_model_from_nvm.py --database_file ${sp_db} --nvm_file slice${slice_num}.nvm --output_dir ${model_dir}

triangulated_dir=triangulated_slice${slice_num}_${ratio}
mkdir ${triangulated_dir}
colmap point_triangulator --database_path ${sp_db} \
  --image_path images/ \
  --input_path ${model_dir}  \
  --output_path ${triangulated_dir} \
