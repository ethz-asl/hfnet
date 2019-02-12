# Update the provided SIFT DB file with the correct intrinsics.
python colmap-helpers/update_db_with_nvm_intrinsics.py --database_file aachen.db --nvm_file aachen_cvpr2018_db.nvm

# Use the provided NVM file for ground-truth poses of all the cameras.
mkdir initial_model_sift
python colmap-helpers/colmap_model_from_nvm.py --database_file aachen.db --nvm_file aachen_cvpr2018_db.nvm --output_dir initial_model_sift/

# Triangulate the 3D model according to the matches provided in the SIFT DB.
mkdir triangulated_model_sift
colmap point_triangulator --database_path aachen.db --image_path images_upright/ --input_path initial_model_sift    --output_path triangulated_model_sift
