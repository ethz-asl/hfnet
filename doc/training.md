# Training with multi-task distillation

## Exporting the target predictions

We first export the predictions of NetVLAD (global descriptor) and SuperPoint (dense keypoint scores and descriptors), which will be the labels of the dataset.

```bash
python3 hfnet/export_predictions.py
	hfnet/configs/netvlad_export_distill.yaml \
	global_descriptors \
	--keys global_descriptor \
	--as_dataset
python3 hfnet/export_predictions.py
	hfnet/configs/superpoint_export_distill.yaml \
	superpoint_predictions \
	--keys local_descriptor_map,dense_scores \
	--as_dataset
```

## Training HF-Net
```bash
python3 hfnet/train.py hfnet/configs/hfnet_train_distill.yaml hfnet
```

The training can be interrupted at any time using `Ctrl+C` and can be monitored with Tensorboard summaries saved in `$EXPER_PATH/hfnet/`. The weights are also saved there.

## Exporting the model for deployment

```bash
python3 hfnet/export_model.py config/hfnet_train_distill.yaml hfnet
```
will export the model to `$EXPER_PATH/saved_models/hfnet/`.
