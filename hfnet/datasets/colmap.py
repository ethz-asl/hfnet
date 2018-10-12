import numpy as np
import tensorflow as tf
from pathlib import Path
import logging
import deepdish
import h5py

from .base_dataset import BaseDataset
from .colmap_utils.read_model import read_model, qvec2rotmat
from hfnet.settings import DATA_PATH


class Colmap(BaseDataset):
    default_config = {
        'truncate': None,
        'sequences': '*',
        'depth_factor': 1.0,
        'make_pairs': False,
        'pair_thresh': 0.2,
        'max_num_pairs': 200,
        'shuffle': False,
        'random_seed': 0,
        'preprocessing': {'resize_max': 640},
        'num_parallel_calls': 10,
    }
    dataset_folder = 'colmap'
    depth_dir = 'depth_maps_clean_300_th_0.10'

    def _init_dataset(self, **config):
        tf.data.Dataset.map_parallel = lambda self, fn: self.map(
            fn, num_parallel_calls=config['num_parallel_calls'])
        base_path = Path(DATA_PATH, self.dataset_folder)

        # Parse sequences
        sequences = config['sequences']
        if not isinstance(sequences, (list, tuple)):
            sequences = [sequences]
        sequences = set([p.stem for s in sequences for p in base_path.glob(s)
                         if p.is_dir()])

        data = {k: [] for k in ['image', 'name', 'depth', 'K']}
        if config['make_pairs']:
            data = {**data, **{k: [] for k in
                    ['image2', 'name2', 'depth2', 'K2', '1_T_2']}}

        for seq in sequences:
            seq_path = Path(base_path, seq)
            cameras, images, points = read_model(
                path=Path(seq_path, 'dense/sparse').as_posix(), ext='.bin')
            image_dir = Path(seq_path, 'dense/images')
            depth_dir = Path(seq_path, 'dense/stereo/'+self.depth_dir)

            # Gather all image info
            data_all = {k: [] for k in ['image', 'name', 'depth', 'K', 'pose']}
            id_mapping = {}
            for i, (id, info) in enumerate(images.items()):
                id_mapping[id] = i
                name = Path(info.name).stem
                K = cameras[id].params
                K = np.array([[K[0], 0, K[2]], [0, K[1], K[3]], [0, 0, 1]])
                T = np.eye(4)
                T[:3, :3] = qvec2rotmat(info.qvec)
                T[:3, 3] = info.tvec

                data_all['name'].append(seq+'/'+name)
                data_all['image'].append(str(Path(image_dir, info.name)))
                data_all['depth'].append(str(Path(depth_dir, name+'.h5')))
                data_all['K'].append(K)
                data_all['pose'].append(T)

            if config['make_pairs']:
                pair_file = next(Path(seq_path, 'dense/stereo').glob('pairs*'))
                pairs = deepdish.io.load(pair_file.as_posix())

                good_pairs = []
                for pair, params in pairs.items():
                    bbox1, bbox2, vis1, vis2, num_matches = params
                    if bbox1 is None or bbox2 is None:
                        continue
                    if np.any(np.array([vis1, vis2]) < config['pair_thresh']):
                        continue
                    good_pairs.append(pair)
                good_pairs = np.array(good_pairs)

                if config['max_num_pairs']:
                    perm = np.random.RandomState(
                        config['random_seed']).permutation(len(good_pairs))
                    good_pairs = good_pairs[perm[:config['max_num_pairs']]]

                for idx1, idx2 in good_pairs:
                    for k in ['image', 'name', 'depth', 'K']:
                        data[k].append(data_all[k][id_mapping[idx1]])
                        data[k+'2'].append(data_all[k][id_mapping[idx2]])
                    T_wto1 = data_all['pose'][id_mapping[idx1]]
                    T_wto2 = data_all['pose'][id_mapping[idx2]]
                    T_2to1 = np.dot(T_wto1, np.linalg.inv(T_wto2))
                    data['1_T_2'].append(T_2to1)
                logging.info('Colmap sequence {} contains {} pairs'.format(
                    seq, len(good_pairs)))
            else:
                logging.info('Colmap sequence {} contains {} images'.format(
                    seq, len(images)))
                for k in data:
                    data[k].extend(data_all[k])

        if config['shuffle']:
            perm = np.random.RandomState(
                config['random_seed']).permutation(len(data['name']))
            data = {k: [v[i] for i in perm] for k, v in data.items()}
        if config['truncate']:
            data = {k: v[:config['truncate']] for k, v in data.items()}
        return data

    def _get_data(self, data, split_name, **config):
        def _read_image(path):
            image = tf.read_file(path)
            image = tf.image.decode_png(image, channels=3)
            return image

        def _py_read_depth(path):
            with h5py.File(path.decode('utf-8'), 'r') as f:
                depth = f['depth'].value
            return (depth*config['depth_factor']).astype(np.float32)

        def _resize_max(image, resize):
            target_size = tf.to_float(tf.convert_to_tensor(resize))
            current_size = tf.to_float(tf.shape(image)[:2])
            scale = target_size / tf.reduce_max(current_size)
            new_size = tf.to_int32(current_size * scale)
            return tf.image.resize_images(
                image, new_size, method=tf.image.ResizeMethod.BILINEAR)

        def _preprocess(data):
            image, depth, K = data['image'], data['depth'], data['K']
            tf.Tensor.set_shape(image, [None, None, 3])
            tf.Tensor.set_shape(depth, [None, None])
            image = tf.image.rgb_to_grayscale(image)
            original_size = tf.shape(image)[:2]
            if config['preprocessing']['resize_max']:
                image = _resize_max(
                    image, config['preprocessing']['resize_max'])
                new_size = tf.shape(image)[:2]
                depth = tf.image.resize_images(
                    depth[..., tf.newaxis], new_size,
                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)[..., 0]
                scale = tf.to_float(new_size) / tf.to_float(original_size)
                scale = tf.diag(tf.concat([scale, [1.]], 0))
                K = tf.matmul(scale, K)
            data['image'] = tf.to_float(image)
            data['depth'] = depth
            data['K'] = K
            return data

        def _parse(data):
            images = tf.data.Dataset.from_tensor_slices(data['image'])
            images = images.map_parallel(_read_image)
            depth = tf.data.Dataset.from_tensor_slices(data['depth'])
            depth = depth.map_parallel(
                lambda path: tf.py_func(_py_read_depth, [path], tf.float32))
            intrinsics = tf.data.Dataset.from_tensor_slices(
                np.array(data['K'], dtype=np.float32))
            names = tf.data.Dataset.from_tensor_slices(data['name'])
            dataset = tf.data.Dataset.zip(
                {'image': images, 'depth': depth, 'K': intrinsics,
                 'name': names})
            dataset = dataset.map_parallel(_preprocess)
            return dataset

        dataset = _parse(data)

        if config['make_pairs']:
            entries = ['image', 'depth', 'K', 'name']
            data2 = {k: data[k+'2'] for k in entries}
            dataset2 = _parse(data2)
            dataset2 = dataset2.map(lambda d: {k+'2': d[k] for k in entries})
            T = tf.data.Dataset.from_tensor_slices(
                np.array(data['1_T_2'], dtype=np.float32))
            dataset = tf.data.Dataset.zip(
                (dataset, dataset2, T)).map(
                    lambda d1, d2, t: {**d1, **d2, '1_T_2': t})

        return dataset
