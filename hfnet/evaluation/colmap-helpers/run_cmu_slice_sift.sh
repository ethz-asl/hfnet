#!/bin/sh

slice_num=$1
slice_db=slice${slice_num}.db

model_dir=model_sift_slice${slice_num}
triangulated_dir=triangulated_sift_slice${slice_num}

rm -rf model_dir
rm -rf triangulated_dir

mkdir ${model_dir}
python colmap-helpers/update_db_with_nvm_intrinsics.py --database_file ${slice_db} --nvm_file slice${slice_num}.nvm
python colmap-helpers/colmap_model_from_nvm.py --database_file ${slice_db} --nvm_file slice${slice_num}.nvm --output_dir ${model_dir}

mkdir ${triangulated_dir}
colmap point_triangulator --database_path ${slice_db} \
  --image_path images/ \
  --input_path ${model_dir}  \
  --output_path ${triangulated_dir} \
