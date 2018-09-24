import numpy as np
import tensorflow as tf
import glob
import random
from pathlib import Path

from .base_dataset import BaseDataset
from hfnet.settings import DATA_PATH


class DescriptorDistillation(BaseDataset):
    default_config = {
        'validation_size': 200,
        'load_descriptors': True,
        'image_folders': [],
        'descriptor_folders': [],
        'truncate': None,
        'shuffle': True,
        'preprocessing': {
            'resize': [480, 640],
            'grayscale': True},
        'cache_in_memory': False,
    }

    def _init_dataset(self, **config):
        assert len(config['image_folders']) > 0
        data = {'names': [], 'image_paths': []}

        if config['load_descriptors']:
            assert len(config['image_folders']) == len(config['descriptor_folders'])
            data['descriptor_paths'] = []

        for i, im_folder in enumerate(config['image_folders']):
            im_paths = sorted(glob.glob(Path(DATA_PATH, im_folder, '*.jpg').as_posix()))
            names = [Path(p).stem for p in im_paths]
            data['names'].extend(names)
            data['image_paths'].extend(im_paths)
            if config['load_descriptors']:
                d_folder = config['descriptor_folders'][i]
                d_paths = [Path(DATA_PATH, d_folder, '{}.npy'.format(n)).as_posix()
                           for n in names]
                data['descriptor_paths'].extend(d_paths)

        data_list = [dict(zip(data, d)) for d in zip(*data.values())]
        if config['shuffle']:
            random.Random(0).shuffle(data_list)
        if config['truncate'] is not None:
            data_list = data_list[:config['truncate']]
        data = {k: [dic[k] for dic in data_list] for k in data_list[0]}
        return data

    def _get_data(self, paths, split_name, **config):
        def _read_image(path):
            image = tf.read_file(path)
            image = tf.image.decode_jpeg(image, channels=3)
            return image

        # Python function
        def _read_descriptor(path):
            return np.load(path.decode('utf-8')).astype(np.float32)

        def _preprocess(image):
            if config['preprocessing']['resize']:
                image = tf.image.resize_images(image, config['preprocessing']['resize'],
                                               method=tf.image.ResizeMethod.BILINEAR)
            if config['preprocessing']['grayscale']:
                image = tf.image.rgb_to_grayscale(image)
            return image

        names = tf.data.Dataset.from_tensor_slices(paths['names'])
        images = tf.data.Dataset.from_tensor_slices(paths['image_paths'])
        images = images.map(_read_image, num_parallel_calls=10)
        images = images.map(_preprocess, num_parallel_calls=10)
        dataset = tf.data.Dataset.zip({'image': images, 'name': names})

        if config['load_descriptors']:
            desc = tf.data.Dataset.from_tensor_slices(paths['descriptor_paths'])
            desc = desc.map(lambda p: tf.py_func(_read_descriptor, [p], tf.float32),
                            num_parallel_calls=10)
            dataset = tf.data.Dataset.zip((dataset, desc)).map(
                    lambda da, de: {**da, 'descriptor': de})

        if split_name == 'validation':
            dataset = dataset.take(config['validation_size'])
        if split_name == 'training':
            dataset = dataset.skip(config['validation_size'])

        if config['cache_in_memory']:
            tf.logging.info('Caching dataset, fist access will take some time.')
            dataset = dataset.cache()

        return dataset
