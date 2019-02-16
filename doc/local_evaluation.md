# Evaluation of local features

As described in our paper, we evaluate several local feature detectors and descriptors on two dataset, [HPatches](https://github.com/hpatches/hpatches-dataset) and [SfM](https://github.com/vcg-uvic/sfm_benchmark_release), following standard metrics for matching and pose estimation.

## Exporting the predictions

```
python3 hfnet/export_predictions.py \
	hfnet/configs/hfnet_export_[hfnet|superpoint|lfnet|doap]_[hpatches|sfm].yaml \
	[hfnet|superpoint|lfnet|doap]/predictions_[hpatches|sfm] \
```

This will export `.npz` files in `$EXPER_PATH/[hfnet|superpoint|lfnet|doap]/predictions_[hpatches|sfm]/`.

## Visualizing the predictions

Head over to `notebooks/visualize_[keypoints|matches]_[hpatches|sfm].ipynb`. The names of the experiments might need to be adjusted.

## Running the evaluation

Head over to `notebooks/evaluation_[detectors|descriptors]_[hpatches|sfm].ipynb`.

