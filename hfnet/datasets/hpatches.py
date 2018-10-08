import numpy as np
import tensorflow as tf
import cv2
from pathlib import Path

from .base_dataset import BaseDataset
from hfnet.settings import DATA_PATH


class Hpatches(BaseDataset):
    default_config = {
        'alteration': 'all',  # 'all', 'i' for illumination, 'v' for viewpoint
        'truncate': None,
        'make_pairs': False,
        'shuffle': False,
        'random_seed': 0,
        'preprocessing': {
            'resize': [480, 640],
        },
        'num_parallel_calls': 10,
    }
    dataset_folder = 'hpatches'
    num_images = 6
    image_ext = '.ppm'

    def _init_dataset(self, **config):
        tf.data.Dataset.map_parallel = lambda self, fn: self.map(
            fn, num_parallel_calls=config['num_parallel_calls'])
        base_path = Path(DATA_PATH, self.dataset_folder)
        scene_paths = sorted([x for x in base_path.iterdir() if x.is_dir()])

        data = {'image': [], 'name': []}
        if config['make_pairs']:
            data = {**data, **{k: [] for k in
                    ['image2', 'name2', 'homography']}}
        for path in scene_paths:
            if config['alteration'] == 'i' and path.stem[0] != 'i':
                continue
            if config['alteration'] == 'v' and path.stem[0] != 'v':
                continue
            for i in range(1, 1 + self.num_images):
                if config['make_pairs']:
                    if i == 1:
                        path2 = str(Path(path, '1' + self.image_ext))
                        name2 = path.stem + '/1'
                        continue
                    data['image2'].append(path2)
                    data['name2'].append(name2)
                    data['homography'].append(
                        np.loadtxt(str(Path(path, 'H_1_' + str(i)))))
                data['image'].append(str(Path(path, str(i) + self.image_ext)))
                data['name'].append(path.stem + '/' + str(i))

        if config['shuffle']:
            perm = np.random.RandomState(
                config['random_seed']).permutation(len(data['name']))
            data = {k: [v[i] for i in perm] for k, v in data.items()}
        if config['truncate']:
            data = {k: v[:config['truncate']] for k, v in data.items()}
        return data

    def _get_data(self, data, split_name, **config):
        def _read_image(path):
            return cv2.imread(path.decode('utf-8'))

        def _resize_max(image, resize):
            target_size = tf.to_float(tf.convert_to_tensor(resize))
            current_size = tf.to_float(tf.shape(image)[:2])
            scale = tf.reduce_max(target_size) / tf.reduce_max(current_size)
            new_size = tf.to_int32(current_size * scale)
            return tf.image.resize_images(
                image, new_size, method=tf.image.ResizeMethod.BILINEAR)

        def _preprocess(image):
            tf.Tensor.set_shape(image, [None, None, 3])
            image = tf.image.rgb_to_grayscale(image)
            original_size = tf.shape(image)[:2]
            if config['preprocessing']['resize']:
                image = _resize_max(
                    image, config['preprocessing']['resize'])
            return tf.to_float(image), original_size

        def _adapt_homography_to_preprocessing(H, data):
            size1 = tf.to_float(tf.shape(data['image'])[:2])
            size2 = tf.to_float(tf.shape(data['image2'])[:2])
            s1 = size1 / tf.to_float(data['original_size'])
            s2 = size2 / tf.to_float(data['original_size2'])
            mult1 = tf.diag(tf.concat([s1, [1.]], 0))
            mult2 = tf.diag(tf.concat([1/s2, [1.]], 0))
            H = tf.matmul(mult1, tf.matmul(tf.to_float(H), mult2))
            return H

        images = tf.data.Dataset.from_tensor_slices(data['image'])
        images = images.map_parallel(
            lambda path: tf.py_func(_read_image, [path], tf.uint8))
        images = images.map_parallel(_preprocess)
        names = tf.data.Dataset.from_tensor_slices(data['name'])
        dataset = tf.data.Dataset.zip(
            (images, names)).map(lambda i, n: {
                'image': i[0], 'original_size': i[1], 'name': n})

        if config['make_pairs']:
            images2 = tf.data.Dataset.from_tensor_slices(data['image2'])
            images2 = images2.map_parallel(
                lambda path: tf.py_func(_read_image, [path], tf.uint8))
            images2 = images2.map_parallel(_preprocess)
            names2 = tf.data.Dataset.from_tensor_slices(data['name2'])
            dataset = tf.data.Dataset.zip(
                (dataset, images2, names2)).map(
                    lambda d, i, n: {
                        'image2': i[0], 'name2': n,
                        'original_size2': i[1], **d})

            homographies = tf.data.Dataset.from_tensor_slices(
                np.array(data['homography']))
            homographies = tf.data.Dataset.zip((homographies, dataset))
            homographies = homographies.map_parallel(
                _adapt_homography_to_preprocessing)
            dataset = tf.data.Dataset.zip(
                (dataset, homographies)).map(
                    lambda d, h: {'homography': h, **d})

        return dataset
