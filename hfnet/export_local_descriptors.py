import numpy as np
import argparse
import yaml
import logging
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
from hfnet.models import get_model  # noqa: E402
from hfnet.datasets import get_dataset  # noqa: E402
from hfnet.utils import tools  # noqa: E402
from hfnet.settings import EXPER_PATH, DATA_PATH  # noqa: E402


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('export_name', type=str)
    parser.add_argument('--exper_name', type=str)
    args = parser.parse_args()

    export_name = args.export_name
    exper_name = args.exper_name
    with open(args.config, 'r') as f:
        config = yaml.load(f)

    base_dir = Path(
        EXPER_PATH, 'exports', ((exper_name+'/') if exper_name else '') + export_name)
    base_dir.mkdir(parents=True, exist_ok=True)

    if exper_name:
        with open(Path(EXPER_PATH, exper_name, 'config.yml'), 'r') as f:
            config = tools.dict_update(yaml.load(f), config)
        checkpoint_path = Path(EXPER_PATH, exper_name)
        if config.get('weights', None):
            checkpoint_path = Path(checkpoint_path, config['weights'])
    else:
        assert 'weights' in config, (
                'Experiment name not found, weights must be provided.')
        checkpoint_path = Path(DATA_PATH, 'weights', config['weights'])

    with get_model(config['model']['name'])(
            data_shape={'image': [None, None, None, config['model']['image_channels']]},
            **config['model']) as net:
        net.load(str(checkpoint_path))
        dataset = get_dataset(config['data']['name'])(**config['data'])
        test_set = dataset.get_test_set()

        pbar = tqdm()
        while True:
            try:
                data = next(test_set)
            except dataset.end_set:
                break
            descriptor = net.predict(data, keys='local_descriptors')

            name = data['name'].decode('utf-8')
            Path(base_dir, Path(name).parent).mkdir(parents=True, exist_ok=True)
            np.save(Path(base_dir, '{}.npy'.format(name)), descriptor)
            pbar.update(1)
        pbar.close()
