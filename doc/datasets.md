# Datasets

All datasets should be downloaded in `$DATA_PATH`. We give below additional details as well as the expected directory structures.

## 6-DoF Localization

These datasets are introduced in [Benchmarking 6DOF Outdoor Visual Localization in Changing Conditions](https://arxiv.org/abs/1707.09092) by Sattler et al., and can be downloaded from [the associated website](http://www.visuallocalization.net/).

For each dataset, there are separate directories for the images, the SfM models, and the localization databases.

#### Aachen Day-Night

```
aachen/
├── aachen.db
├── query.db
├── day_time_queries_with_intrinsics.txt
├── night_time_queries_with_intrinsics.txt
├── databases/
├── images_upright/
│   ├── db/
│   │   └── ...
│   └── query/
│       └── ...
└── models/
    └── hfnet_model/
        ├── cameras.bin
        ├── images.bin
        └── points3D.bin
```

#### RobotCar Seasons

```
robotcar/
├── overcast-reference.db
├── query.db
├── images/
│   ├── overcast-reference/
│   ├── sun/
│   ├── dusk/
│   ├── night/
│   └── night-rain/
├── intrinsics/
│   ├── left_intrinsics.txt
│   ├── rear_intrinsics.txt
│   └── right_intrinsics.txt
├── queries/
│   ├── dusk_queries_with_intrinsics.txt
│   ├── night_queries_with_intrinsics.txt
│   ├── night-rain_queries_with_intrinsics.txt
│   └── sun_queries_with_intrinsics.txt
└── models/
    └── hfnet_model/
        ├── cameras.bin
        ├── images.bin
        └── points3D.bin
```
The query lists generated with `setup/utils/generate_robotcar_query_list.py` are available [here](http://robotics.ethz.ch/~asl-datasets/2019_CVPR_hierarchical_localization/query_lists_robotcar.tar.gz).

#### CMU Seasons

```
cmu/
├── images/
│   ├── slice2/
│   │   ├── database/
│   │   └── query/
│   └── ...
├── slice2/
│   ├── sift_database.db
│   ├── sift_queries.db
│   ├── slice2.queries_with_intrinsics.txt
│   └── models/
│       └── hfnet_model/
│           ├── cameras.bin
│           ├── images.bin
│           └── points3D.bin
├── slice3/
│   └── ...
└── ...
```

## Local features evaluation

Local feature detectors and descriptors can be evaluated on the HPatches and SfM datasets, as reported in our paper.

#### HPatches

The dataset is described in the paper [HPatches: A benchmark and evaluation of handcrafted and learned local descriptors](https://arxiv.org/pdf/1704.05939.pdf) by Balntas et al., and can be downloaded [here](https://github.com/hpatches/hpatches-dataset).

```
hpatches/
├── i_ajuntament/
└── ...
```

#### SfM

The dataset is introduced by Ono et al. in their paper [LF-Net: Learning Local Features from Images and can be obtained [here](https://github.com/vcg-uvic/sfm_benchmark_release).

```
sfm/
├── british_museum/
│   └── dense/
├── florence_cathedral_side/
│   └── dense/
├── lincoln_memorial_statue/
│   └── dense/
├── london_bridge/
│   └── dense/
├── milan_cathedral/
│   └── dense/
├── mount_rushmore/
│   └── dense/
├── piazza_san_marco/
│   └── dense/
├── reichstag/
│   └── dense/
├── sacre_coeur/
│   └── dense/
├── sagrada_familia/
│   └── dense/
├── st_pauls_cathedral/
│   └── dense/
├── united_states_capitol/
│   └── dense/
├── scales.txt
└── exif
    ├── brandenburg_gate
    ├── british_museum
    ├── buckingham
    ├── colosseum_exterior
    ├── florence_cathedral_side
    ├── grand_place_brussels
    ├── hagia_sophia_interior
    ├── lincoln_memorial_statue
    ├── london_bridge
    ├── milan_cathedral
    ├── mount_rushmore
    ├── notre_dame_front_facade
    ├── pantheon_exterior
    ├── piazza_san_marco
    ├── sagrada_familia
    ├── st_pauls_cathedral
    ├── st_peters_square
    ├── taj_mahal
    ├── temple_nara_japan
    ├── trevi_fountain
    ├── united_states_capitol
    └── westminster_abbey
```

## Multi-task distillation

HF-Net is trained on the Google Landmarks and Berkeley Deep Drive datasets. For the former, first download the [index of images](https://github.com/ethz-asl/hierarchical_loc/releases/download/1.0/google_landmarks_index.csv) and then the dataset itself using the script `setup/scripts/download_google_landmarks.py`. The latter can be downloaded on the [dataset website](https://bdd-data.berkeley.edu/).

The labels are predictions of SuperPoint and NetVLAD. Their export is described in the [training documentation](doc/training.md).

```
google_landmarks/
├── images/
├── global_descriptors/
└── superpoint_predictions/
bdd/
├── dawn_images_vga/
├── night_images_vga/
├── global_descriptors/
└── superpoint_predictions/
```
