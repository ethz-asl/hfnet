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
        'preprocessing': {
            'resize': [480, 640],
        }
    }
    dataset_folder = 'hpatches'
    num_images = 5
    image_ext = '.ppm'

    def _init_dataset(self, **config):
        base_path = Path(DATA_PATH, self.dataset_folder)
        scene_paths = sorted([x for x in base_path.iterdir() if x.is_dir()])

        image_paths = []
        warped_image_paths = []
        homographies = []
        names = []
        for path in scene_paths:
            if config['alteration'] == 'i' and path.stem[0] != 'i':
                continue
            if config['alteration'] == 'v' and path.stem[0] != 'v':
                continue
            ref_path = str(Path(path, '1' + self.image_ext))
            for i in range(2, 2 + self.num_images):
                image_paths.append(ref_path)
                warped_image_paths.append(str(Path(path, str(i) + self.image_ext)))
                homographies.append(np.loadtxt(str(Path(path, 'H_1_' + str(i)))))
                names.append(path.stem + '/' + str(i))
        data = {'image_paths': image_paths,
                'warped_image_paths': warped_image_paths,
                'homographies': homographies,
                'names': names}

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
        homographies = tf.data.Dataset.from_tensor_slices(np.array(data['homographies']))
        if config['preprocessing']['resize']:
            homographies = tf.data.Dataset.zip({'image': images,
                                                'homography': homographies})
            homographies = homographies.map(_adapt_homography_to_preprocessing)
        images = images.map(_preprocess)

        warped_images = tf.data.Dataset.from_tensor_slices(data['warped_image_paths'])
        warped_images = warped_images.map(lambda path: tf.py_func(_read_image,
                                                                  [path],
                                                                  tf.uint8))
        warped_images = warped_images.map(_preprocess)

        names = tf.data.Dataset.from_tensor_slices(data['names'])

        data = tf.data.Dataset.zip({'image': images, 'warped_image': warped_images,
                                    'homography': homographies, 'name': names})
        return data
