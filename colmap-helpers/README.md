# Building SfM models

We provide here scripts to build SfM models for the Aachen, RobotCar, and CMU datasets using [COLMAP](https://colmap.github.io/) and any learned features. While we used SuperPoint keypoints and descriptors, this process should work for any reasonable feature predictor.

## Exporting dense features

If not already done when evaluating the localization, we first export the network predictions, e.g. dense keypoint scores and descriptors:
```bash
python3 hfnet/export_predictions.py \
	hfnet/configs/hfnet_export_[aachen|cmu|robotcar]_db.yaml \
	[aachen|cmu|robotcar] \
	--exper_name hfnet \
	--keys keypoints,scores,local_descriptor_map
```

This will create an `.npz` file in `$EXPER_PATH/exports/hfnet/[aachen|cmu|robotcar]/` for each database image.

## Extracting sparse features

For increased flexibility, we only subsequently extract features using non-maximum suppression (NMS) and bilinear interpolation:
```bash
python3 colmap-helpers/export_for_sfm.py \
	[aachen|cmu|robotcar] \  # dataset name
	hfnet/[aachen|cmu|robotcar] \  # input directory
	hfnet/[aachen|cmu|robotcar]_sfm  # output directory
```
This creates new `.npz` files in `$EXPER_PATH/exports/hfnet/[aachen|cmu|robotcar]_sfm/`. Parameters for extraction, such as the NMS radius or the number of keypoints, can be adjusted in `colmap-helpers/export_for_sfm.py`. Why this complicated process? We want to keep dense predictions accessible on disk so as to experiment with the extraction parameters when evaluating the localization (e.g. impact of the number of keypoints).

## Building the model

Assuming we have reference models that contain accurate poses (e.g. from SIFT, as provided by the benchmark authors), we can match keypoints between images and triangulate the 3D points using COLMAP. This is much faster than optimizing all the poses again with bundle adjunstment, and ensures that the estimated query poses are consistent across models.

The process goes as follows:
- Export `.txt` files containing keypoint locations with `features_from_npz.py`.
- Match features accross frames with `match_features_with_db_prior.py`. To speed up the process, we use the original SIFT database file to only match frames that are covisible in the SIFT model.
- Generate a new database file with COLMAP commands `database_creator`, `feature_importer`, and `matches_importer`.
- Import camera poses from a SIFT NVM model file by creating a dummy intermediate model with `colmap_model_from_nvm.py`.
- Triangulate the new model with the COLMAP command `point_triangulator`.

As there are some slight variations between the datasets (e.g. importing intrinsics from the SIFT database or from an NVM model), we provide **reference** scripts in `example_scripts/`. Paths might need to be adjusted to work in your workspace.
