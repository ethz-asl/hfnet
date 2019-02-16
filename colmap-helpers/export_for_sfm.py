import logging
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np

logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
from hfnet.datasets import get_dataset  # noqa: E402
from hfnet.evaluation.loaders import export_loader  # noqa: E402
from hfnet.settings import EXPER_PATH  # noqa: E402


def export_for_sfm(data_config, exper_config, export_name):
    export_dir = Path(EXPER_PATH, 'exports', export_name)
    export_dir.mkdir(exist_ok=True)

    dataset = get_dataset(data_config['name'])(**data_config)
    data_iter = dataset.get_test_set()

    for data in tqdm(data_iter):
        predictions = exper_config['predictor'](
            data['image'], data['name'], **exper_config)
        # Scale the keypoints to the original image size
        # and convert to Colmap convention (origin = corner of upper left pix)
        scale = ((np.array(data['original_size'][:2]) - 1)
                 / (np.array(data['image'].shape[:2]) - 1))
        export = {
            'keypoints': scale[::-1] * predictions['keypoints'] + 0.5,
            'scores': predictions['scores'],
            'descriptors': predictions['descriptors'],
            'image_size': data['image'].shape[:2][::-1]
        }
        name = data['name'].decode('utf-8')
        Path(export_dir, Path(name).parent).mkdir(parents=True, exist_ok=True)
        np.savez(Path(export_dir, f'{name}.npz'), **export)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str)
    parser.add_argument('export_name', type=str)
    parser.add_argument('new_export_name', type=str)
    args = parser.parse_args()

    data_configs = {
        'aachen': {
            'name': 'aachen',
            'load_db': True,
            'load_queries': False,
            'resize_max': 960,
        },
        'robotcar': {
            'name': 'robotcar',
            'sequences': ['overcast-reference'],
            'resize_max': 960,
        },
        'cmu': {
            'name': 'cmu',
            'sequences': ['slice2', 'slice3', 'slice4', 'slice5', 'slice6',
                          'slice7', 'slice8', 'slice9', 'slice10', 'slice17'],
            'load_db': True,
            'load_queries': False,
            'resize_max': 1024,
        }
    }

    exper_config = {
        'experiment': args.export_name,
        'predictor': export_loader,
        'num_features': 3000,
        'do_nms': True,
        'nms_thresh': 4,
    }

    export_for_sfm(
        data_configs[args.dataset], exper_config, args.new_export_name)
