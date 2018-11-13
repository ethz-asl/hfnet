import numpy as np
import tensorflow as tf
import logging
import random
from pathlib import Path

from .base_dataset import BaseDataset
from .utils import pipeline
from hfnet.settings import DATA_PATH


tf.data.Dataset.map_parallel = lambda self, fn: self.map(
    fn, num_parallel_calls=10)
tf.data.Dataset.keys = lambda self: list(self.output_types.keys())


class Distillation(BaseDataset):
    default_config = {
        'validation_size': 200,
        'load_targets': True,
        'image_dirs': [],
        'targets': [],
        'truncate': None,
        'shuffle': True,
        'preprocessing': {
            'resize': [480, 640],
            'grayscale': True},
        'cache_in_memory': False,
        'augmentation': {
            'photometric': {
                'enable': False,
                'primitives': 'all',
                'params': {},
                'random_order': True,
            },
            'homographic': {
                'enable': False,
                'params': {},
                'valid_border_margin': 0,
            },
        },
        'for_batching': True,
    }

    def _init_dataset(self, **config):
        data = {'names': [], 'images': []}
        if config['load_targets']:
            for i, target in enumerate(config['targets']):
                for im in config['image_dirs']:
                    assert Path(Path(DATA_PATH, im).parent,
                                target['dir']).exists()
                data[i] = []

        logging.info('Listing image files')
        im_paths = []
        names = []
        for i, image_dir in enumerate(config['image_dirs']):
            paths = Path(DATA_PATH, image_dir).glob('*.jpg')
            paths = sorted([str(p) for p in paths])
            if config['truncate'] is not None:
                t = config['truncate'][i]
                if t is not None:
                    paths = paths[:t]
            im_paths.extend(paths)
            names.extend([Path(p).stem for p in paths])

        if config['load_targets']:
            logging.info('Listing target files')
            for im, n in zip(im_paths, names):
                ok = True
                target_paths = []
                for target in config['targets']:
                    target_path = Path(
                        Path(im).parent.parent, target['dir'], f'{n}.npz')
                    # target_path = Path(DATA_PATH, target['dir'], f'{n}.npz')
                    ok &= target_path.exists()
                    target_paths.append(target_path.as_posix())
                if not ok:
                    continue
                data['images'].append(im)
                data['names'].append(n)
                for i, p in enumerate(target_paths):
                    data[i].append(p)
        else:
            data['names'].extend(names)
            data['images'].extend(im_paths)

        data_list = [dict(zip(data, d)) for d in zip(*data.values())]
        if config['shuffle']:
            random.Random(0).shuffle(data_list)
        data = {k: [dic[k] for dic in data_list] for k in data_list[0]}
        logging.info(f'Dataset size: {len(data["images"])}')
        return data

    def _get_data(self, paths, split_name, **config):
        is_training = split_name == 'training'

        def _read_image(path):
            image = tf.read_file(path)
            image = tf.image.decode_jpeg(image, channels=3)
            return image

        # Python function
        def _create_npz_reader(keys):
            def _read_npz(keys, path):
                npz = np.load(path.decode('utf-8'))
                return [npz[k].astype(np.float32) for k in keys]
            return lambda x: _read_npz(keys, x)

        def _preprocess(image):
            if config['preprocessing']['resize']:
                image = tf.image.resize_images(
                        image, config['preprocessing']['resize'],
                        method=tf.image.ResizeMethod.BILINEAR)
            if config['preprocessing']['grayscale']:
                image = tf.image.rgb_to_grayscale(image)
            return image

        def _delete_keys(data):
            keys = ['keypoints']
            for k in keys:
                data.pop(k, None)
            return data

        names = tf.data.Dataset.from_tensor_slices(paths['names'])
        images = tf.data.Dataset.from_tensor_slices(paths['images'])
        images = images.map_parallel(_read_image)
        images = images.map_parallel(_preprocess)
        dataset = tf.data.Dataset.zip({'image': images, 'name': names})

        if config['load_targets']:
            for i, target in enumerate(config['targets']):
                t = tf.data.Dataset.from_tensor_slices(paths[i])
                reader = _create_npz_reader(target['keys'])
                types = [tf.float32]*len(target['keys'])
                t = t.map_parallel(lambda p: tf.py_func(reader, [p], types))
                dataset = tf.data.Dataset.zip((dataset, t)).map(
                    lambda da, de: {**da, **{k: de[j]
                                    for j, k in enumerate(target['keys'])}})

            # Reversed convention...
            if 'keypoints' in dataset.keys():
                dataset = dataset.map(
                    lambda d: {
                        **d, 'keypoints': tf.reshape(
                            d['keypoints'][:, ::-1], [-1, 2])})

        if split_name == 'validation':
            dataset = dataset.take(config['validation_size'])
        if split_name == 'training':
            dataset = dataset.skip(config['validation_size'])

        if config['cache_in_memory']:
            tf.logging.info('Caching dataset, fist access will take some time')
            dataset = dataset.cache()

        if is_training:
            if config['augmentation']['photometric']['enable']:
                dataset = dataset.map_parallel(
                    lambda d: pipeline.photometric_augmentation(
                        d, **config['augmentation']['photometric']))
            if config['augmentation']['homographic']['enable']:
                dataset = dataset.map_parallel(
                    lambda d: pipeline.homographic_augmentation(
                        d, **config['augmentation']['homographic']))

        if 'keypoints' in dataset.keys():
            dataset = dataset.map_parallel(pipeline.add_keypoint_map)
        if config['for_batching']:
            dataset = dataset.map_parallel(_delete_keys)

        return dataset
