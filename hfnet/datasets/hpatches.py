import numpy as np
import tensorflow as tf
import cv2
from pathlib import Path

from .base_dataset import BaseDataset
from hfnet.settings import DATA_PATH


class Hpatches(BaseDataset):
    default_config = {
        'alteration': 'all',  # 'all', 'i' for illumination or 'v' for viewpoint
        'truncate': None,
        'make_pairs': False,
        'shuffle': False,
        'random_seed': 0,
        'preprocessing': {
            'resize': [480, 640],
        }
    }
    dataset_folder = 'hpatches'
    num_images = 6
    image_ext = '.ppm'

    def _init_dataset(self, **config):
        base_path = Path(DATA_PATH, self.dataset_folder)
        scene_paths = sorted([x for x in base_path.iterdir() if x.is_dir()])

        data = {'image_paths': [], 'names': []}
        if config['make_pairs']:
            data = {**data, **{k: [] for k in
                    ['ref_paths', 'ref_names', 'homographies']}}
        for path in scene_paths:
            if config['alteration'] == 'i' and path.stem[0] != 'i':
                continue
            if config['alteration'] == 'v' and path.stem[0] != 'v':
                continue
            for i in range(1, 1 + self.num_images):
                if config['make_pairs']:
                    if i == 1:
                        ref_path = str(Path(path, '1' + self.image_ext))
                        ref_name = path.stem + '/1'
                        continue
                    data['ref_paths'].append(ref_path)
                    data['ref_names'].append(ref_name)
                    data['homographies'].append(
                        np.loadtxt(str(Path(path, 'H_1_' + str(i)))))
                data['image_paths'].append(
                    str(Path(path, str(i) + self.image_ext)))
                data['names'].append(path.stem + '/' + str(i))

        if config['shuffle']:
            perm = np.random.RandomState(
                config['random_seed']).permutation(len(data['names']))
            data = {k: [v[i] for i in perm] for k, v in data.items()}
        if config['truncate']:
            data = {k: v[:config['truncate']] for k, v in data.items()}
        return data

    def _get_data(self, data, split_name, **config):
        def _read_image(path):
            return cv2.imread(path.decode('utf-8'))

        def _ratio_preserving_resize(image, **config):
            target_size = tf.convert_to_tensor(config['resize'])
            scales = tf.to_float(tf.divide(target_size, tf.shape(image)[:2]))
            new_size = tf.to_float(tf.shape(image)[:2]) * tf.reduce_max(scales)
            image = tf.image.resize_images(image, tf.to_int32(new_size),
                                           method=tf.image.ResizeMethod.BILINEAR)
            return tf.image.resize_image_with_crop_or_pad(image, target_size[0],
                                                          target_size[1])

        def _preprocess(image):
            tf.Tensor.set_shape(image, [None, None, 3])
            image = tf.image.rgb_to_grayscale(image)
            if config['preprocessing']['resize']:
                image = _ratio_preserving_resize(image, **config['preprocessing'])
            return tf.to_float(image)

        def _adapt_homography_to_preprocessing(zip_data):
            image = zip_data['image']
            H = tf.cast(zip_data['homography'], tf.float32)
            target_size = tf.convert_to_tensor(config['preprocessing']['resize'])
            s = tf.reduce_max(tf.cast(tf.divide(target_size,
                                                tf.shape(image)[:2]), tf.float32))
            down_scale = tf.diag(tf.stack([1/s, 1/s, tf.constant(1.)]))
            up_scale = tf.diag(tf.stack([s, s, tf.constant(1.)]))
            H = tf.matmul(up_scale, tf.matmul(H, down_scale))
            return H

        images = tf.data.Dataset.from_tensor_slices(data['image_paths'])
        images = images.map(lambda path: tf.py_func(_read_image, [path], tf.uint8))
        images = images.map(_preprocess)
        names = tf.data.Dataset.from_tensor_slices(data['names'])
        dataset = tf.data.Dataset.zip({'image': images, 'name': names})

        if config['make_pairs']:
            homographies = tf.data.Dataset.from_tensor_slices(
                np.array(data['homographies']))
            if config['preprocessing']['resize']:
                homographies = tf.data.Dataset.zip(
                    {'image': images, 'homography': homographies})
                homographies = homographies.map(
                    _adapt_homography_to_preprocessing)
            images_ref = tf.data.Dataset.from_tensor_slices(data['ref_paths'])
            images_ref = images_ref.map(
                lambda path: tf.py_func(_read_image, [path], tf.uint8))
            images_ref = images_ref.map(_preprocess)
            names_ref = tf.data.Dataset.from_tensor_slices(data['ref_names'])
            # dataset = tf.data.Dataset.zip(
                # {'image_ref': images_ref, 'name_ref': names_ref,
                 # 'homography': homographies, **dataset})
            dataset = tf.data.Dataset.zip(
                (dataset, images_ref, names_ref, homographies)).map(
                    lambda d, i, n, h: {
                        'image_ref': i, 'name_ref': n,
                        'homography': h, **d})
        return dataset
