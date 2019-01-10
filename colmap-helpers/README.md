## Building a Colmap model using custom features and matches ##


### Feature detection and extraction ###
If you decide to use custom features, we first generate the feature detection/extraction files (one per image) from NPZ Numpy files using the following script:
```
python colmap-helpers/features_from_npz.py --npz_dir <npz_directory> --image_dir <image_directory>
```
example:
```
python colmap-helpers/features_from_npz.py --npz_dir db --image_dir images_upright/db/
```


### Feature matching ###
Then, we need to match features across the frames. To speed up this process, we can use the original db file (e.g. using SIFT) to only match the pairs of frames that were matching well with original features.
```
python colmap-helpers/match_features_with_db_prior.py --database_file <db_file> --image_prefix db --image_dir <folder_with_images> --output_file <output_file> --min_num_matches=<min_num_matches_in_db> --num_points_per_frame=<points_per_frame>
```
example:
```
python colmap-helpers/match_features_with_db_prior.py --database_file aachen.db --image_prefix db --image_dir images_upright --output_file matches.txt --min_num_matches=15 --num_points_per_frame=2000
```
You can also use ``--debug`` flag for additional debugging.


### Generating the Colmap db file ###
Next, we need to generate the new Colmap db file with our features and matches:
```
colmap database_creator --database_path database.db
colmap feature_importer --database_path database.db --image_path images_upright/ --import_path images_upright/
colmap matches_importer --database_path database.db --match_list_path matches.txt --match_type raw
```


### Using prior image poses ###
We could now run the sparse reconstruction of Colmap, but we would actually like to reuse the ground-truth database frame poses from the original poses. We therefore provide a script that reads an existing ground-truth NVM model and uses the camera poses to triangulate the 3D points according to custom matches as imported above.
```
python colmap-helpers/colmap_model_from_nvm.py --database_file <db_file> --nvm_file <nvm_file> --output_dir <model_output_directory>
```
example:
```
python colmap-helpers/colmap_model_from_nvm.py --database_file aachen.db --nvm_file aachen_cvpr2018_db.nvm --output_dir new_model/
```


### Model triangulation ###
Finally, you can triangulate the new model using the standard colmap command:
```
colmap point_triangulator --database_path manual.db --image_path <image_directory> --input_path input_model     --output_path triangulated_model
```
