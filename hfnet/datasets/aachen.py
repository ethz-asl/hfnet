import tensorflow as tf
from pathlib import Path

from .base_dataset import BaseDataset
from hfnet.settings import DATA_PATH


class Aachen(BaseDataset):
    default_config = {
        'load_db': True,
        'load_queries': True,
        'resize_max': 640,
        'num_parallel_calls': 10,
    }
    dataset_folder = 'aachen/images_upright'

    def _init_dataset(self, **config):
        tf.data.Dataset.map_parallel = lambda self, fn: self.map(
            fn, num_parallel_calls=config['num_parallel_calls'])
        base_path = Path(DATA_PATH, self.dataset_folder)

        search = []
        if config['load_db']:
            search.append(Path(base_path, 'db'))
        if config['load_queries']:
            search.append(Path(base_path, 'query'))
        assert len(search) != 0

        data = {'image': [], 'name': []}
        for s in search:
            for p in s.glob('**/*.jpg'):
                data['image'].append(p.as_posix())
                rel = p.relative_to(base_path)
                data['name'].append(Path(rel.parent, rel.stem).as_posix())
        return data

    def _get_data(self, data, split_name, **config):
        def _read_image(path):
            image = tf.read_file(path)
            image = tf.image.decode_png(image, channels=3)
            return image

        def _resize_max(image, resize):
            target_size = tf.to_float(tf.convert_to_tensor(resize))
            current_size = tf.to_float(tf.shape(image)[:2])
            scale = target_size / tf.reduce_max(current_size)
            new_size = tf.to_int32(current_size * scale)
            return tf.image.resize_images(
                image, new_size, method=tf.image.ResizeMethod.BILINEAR)

        def _preprocess(image):
            tf.Tensor.set_shape(image, [None, None, 3])
            image = tf.image.rgb_to_grayscale(image)
            if config['resize_max']:
                image = _resize_max(image, config['resize_max'])
            return tf.to_float(image)

        images = tf.data.Dataset.from_tensor_slices(data['image'])
        images = images.map_parallel(_read_image)
        images = images.map_parallel(_preprocess)
        names = tf.data.Dataset.from_tensor_slices(data['name'])
        dataset = tf.data.Dataset.zip({'image': images, 'name': names})
        return dataset
