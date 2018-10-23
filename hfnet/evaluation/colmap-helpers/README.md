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
python colmap-helpers/match_features_with_db_prior.py --database_file <db_file> --filter_image_dir <>
```
example:
```

```

### Generating the Colmap db file ###
Next, we need to generate the new Colmap db file with our features and matches:
```
blah
```

### Using ###
We could now run the sparse reconstruction of Colmap, but we would actually like to reuse the ground-truth database frame poses from the original poses. We therefore provide a script that


### Model triangulation ###
Finally, you can triangulate the new model using the standard colmap command:
```
colmap point_triangulator --database_path manual.db --image_path <image_directory> --input_path input_model     --output_path triangulated_model
```
