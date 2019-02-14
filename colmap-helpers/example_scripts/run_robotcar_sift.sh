# Fix the inconsistency in the image file extension. The database fill be updated to JPG file paths.
python3 colmap-helpers/robotcar_db_png_to_jpg.py --db_file overcast-reference.db

# Update the provided SIFT DB file with the correct intrinsics.
python3 colmap-helpers/update_db_with_nvm_intrinsics.py --database_file overcast-reference.db --nvm_file all.nvm

# Use the provided NVM file for ground-truth poses of all the cameras.
mkdir initial_model_sift_robotcar
python3 colmap-helpers/colmap_model_from_nvm.py --database_file overcast-reference.db --nvm_file all.nvm --output_dir initial_model_sift_robotcar/

# Triangulate the 3D model according to the matches provided in the SIFT DB.
mkdir triangulated_model_sift_robotcar/
colmap point_triangulator --database_path overcast-reference.db --image_path . --input_path initial_model_sift_robotcar    --output_path triangulated_model_sift_robotcar
