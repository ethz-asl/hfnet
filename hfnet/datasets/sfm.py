import numpy as np
import tensorflow as tf
from pathlib import Path
import logging
import deepdish
import h5py
from math import cos, sin, radians

from .base_dataset import BaseDataset
from .colmap_utils.read_model import read_model, qvec2rotmat
from hfnet.settings import DATA_PATH


rotation_matrices = tf.constant(
    [[[cos(r), -sin(r), 0., 0.], [sin(r), cos(r), 0., 0.],
      [0., 0., 1., 0.], [0., 0., 0., 1.]]
     for r in [radians(d) for d in (0, 270, 180, 90)]])


class Sfm(BaseDataset):
    default_config = {
        'truncate': None,
        'sequences': '*',
        'make_pairs': False,
        'pair_thresh': 0.2,
        'max_num_pairs': 200,
        'shuffle': False,
        'random_seed': 0,
        'preprocessing': {
            'resize_max': 640,
            'grayscale': True,
            'upright': False,
        },
        'scale_file': 'scales.txt',
        'num_parallel_calls': 10,
    }
    dataset_folder = 'sfm'
    depth_dir = 'depth_maps_clean_300_th_0.10'
    exif_dir = 'exif'
    exif_to_rot = {
        'TopLeft': 0, 'BottomRight': 2, 'RightTop': 3, 'LeftBottom': 1,
        'Undefined': 0,
    }

    def _parse_exif(self, p):
        if p.exists():
            with open(p, 'r', errors='ignore') as f:
                for line in f:
                    if '  Orientation:' in line:
                        rot = line.split()[1]
                        return self.exif_to_rot[rot]
                logging.info('Rotation not found in EXIF')
        else:
            logging.info('EXIF file not found: %s', p)
        return 0

    def _init_dataset(self, **config):
        tf.data.Dataset.map_parallel = lambda self, fn: self.map(
            fn, num_parallel_calls=config['num_parallel_calls'])
        base_path = Path(DATA_PATH, self.dataset_folder)

        # Parse sequences
        sequences = config['sequences']
        if not isinstance(sequences, (list, tuple)):
            sequences = [sequences]
        sequences = sorted(set([p.stem for s in sequences
                                for p in base_path.glob(s) if p.is_dir()]))

        # Read scale file
        scales = {}
        scale_file = config['scale_file']
        if scale_file and Path(base_path, scale_file).exists():
            with open(Path(base_path, scale_file).as_posix(), 'r') as f:
                for line in f.readlines():
                    seq, scale = line.split()
                    scales[seq] = float(scale)

        data = {k: [] for k in ['image', 'name', 'depth', 'K', 'scale', 'rot']}
        if config['make_pairs']:
            data = {**data, **{k: [] for k in
                    ['image2', 'name2', 'depth2', 'K2', '1_T_2', 'rot2']}}
        else:
            data = {**data, 'i_T_w': []}

        for seq in sequences:
            seq_path = Path(base_path, seq)
            cameras, images, points = read_model(
                path=Path(seq_path, 'dense/sparse').as_posix(), ext='.bin')
            image_dir = Path(seq_path, 'dense/images')
            depth_dir = Path(seq_path, 'dense/stereo/'+self.depth_dir)
            scale = scales.get(seq, 1.0)

            # Gather all image info
            data_all = {k: [] for k in ['image', 'name', 'depth',
                                        'K', 'i_T_w', 'scale', 'rot']}
            id_mapping = {}
            for i, (id, info) in enumerate(images.items()):
                id_mapping[id] = i
                name = Path(info.name).stem
                K = cameras[id].params
                K = np.array([[K[0], 0, K[2]-0.5],
                              [0, K[1], K[3]-0.5],
                              [0, 0, 1]])
                T = np.eye(4)
                T[:3, :3] = qvec2rotmat(info.qvec)
                T[:3, 3] = info.tvec*scale

                if config['preprocessing']['upright']:
                    exif_p = Path(base_path, self.exif_dir, seq)
                    if not exif_p.exists():
                        if i == 0:
                            logging.info('No EXIF data for sequence %s', seq)
                        rot = 0
                    else:
                        rot = self._parse_exif(Path(exif_p, info.name+'.txt'))
                else:
                    rot = 0

                data_all['name'].append(seq+'/'+name)
                data_all['image'].append(str(Path(image_dir, info.name)))
                data_all['depth'].append(str(Path(depth_dir, name+'.h5')))
                data_all['K'].append(K)
                data_all['i_T_w'].append(T)
                data_all['scale'].append(scale)
                data_all['rot'].append(rot)

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
                    for k in ['image', 'name', 'depth', 'K', 'rot']:
                        data[k].append(data_all[k][id_mapping[idx1]])
                        data[k+'2'].append(data_all[k][id_mapping[idx2]])
                    T_wto1 = data_all['i_T_w'][id_mapping[idx1]]
                    T_wto2 = data_all['i_T_w'][id_mapping[idx2]]
                    T_2to1 = np.dot(T_wto1, np.linalg.inv(T_wto2))
                    data['1_T_2'].append(T_2to1)
                    data['scale'].append(scale)
                logging.info('SfM sequence {} contains {} pairs'.format(
                    seq, len(good_pairs)))
            else:
                logging.info('SfM sequence {} contains {} images'.format(
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

        def _py_read_depth(path, scale):
            with h5py.File(path.decode('utf-8'), 'r') as f:
                depth = f['depth'].value
            return (depth*scale).astype(np.float32)

        def _resize_max(image, resize):
            target_size = tf.to_float(tf.convert_to_tensor(resize))
            current_size = tf.to_float(tf.shape(image)[:2])
            scale = target_size / tf.reduce_max(current_size)
            new_size = tf.to_int32(current_size * scale)
            return tf.image.resize_images(
                image, new_size, method=tf.image.ResizeMethod.BILINEAR)

        # TODO: fix principal point coordinates
        def _rotate_intrinsics(K, image, rot):
            shape = tf.to_float(tf.shape(image))
            h, w = shape[0], shape[1]
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            def r0():  # noqa E306
                return K
            def r1():  # noqa E306
                return tf.convert_to_tensor([[fy, 0., cy],
                                             [0., fx, w-1-cx],
                                             [0., 0., 1.]])
            def r2():  # noqa E306
                return tf.convert_to_tensor([[fx, 0., w-1-cx],
                                             [0., fy, h-1-cy],
                                             [0., 0., 1.]])
            def r3():  # noqa E306
                return tf.convert_to_tensor([[fy, 0., h-1-cy],
                                             [0., fx, cx],
                                             [0., 0., 1.]])

            return tf.case([(tf.equal(rot, i), f)
                            for i, f in enumerate([r0, r1, r2, r3])])

        def _rotate_extrinsics(T_2to1, rot1, rot2):
            T = tf.matmul(rotation_matrices[rot1], T_2to1)
            return tf.matmul(T, tf.linalg.inv(rotation_matrices[rot2]))

        def _preprocess(data):
            image, depth, K = data['image'], data['depth'], data['K']
            tf.Tensor.set_shape(image, [None, None, 3])
            tf.Tensor.set_shape(depth, [None, None])
            if config['preprocessing']['grayscale']:
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
            if config['preprocessing']['upright']:
                rot = data['rot']
                K = _rotate_intrinsics(K, image, rot)
                image = tf.image.rot90(image, rot)
                depth = tf.image.rot90(depth[..., None], rot)[..., 0]
            data['image'] = tf.to_float(image)
            data['depth'] = depth
            data['K'] = K
            return data

        def _parse(data):
            images = tf.data.Dataset.from_tensor_slices(data['image'])
            images = images.map_parallel(_read_image)
            depth = tf.data.Dataset.from_tensor_slices(data['depth'])
            scales = tf.data.Dataset.from_tensor_slices(data['scale'])
            rotations = tf.data.Dataset.from_tensor_slices(data['rot'])
            depth = tf.data.Dataset.zip((depth, scales)).map_parallel(
                lambda d, s: tf.py_func(_py_read_depth, [d, s], tf.float32))
            intrinsics = tf.data.Dataset.from_tensor_slices(
                np.array(data['K'], dtype=np.float32))
            names = tf.data.Dataset.from_tensor_slices(data['name'])
            dataset = tf.data.Dataset.zip(
                {'image': images, 'depth': depth, 'K': intrinsics,
                 'rot': rotations, 'name': names})
            dataset = dataset.map_parallel(_preprocess)
            return dataset

        dataset = _parse(data)

        if config['make_pairs']:
            entries = ['image', 'depth', 'K', 'name', 'rot']
            data2 = {k: data[k+'2'] for k in entries}
            data2['scale'] = data['scale']
            dataset2 = _parse(data2)
            dataset2 = dataset2.map(lambda d: {k+'2': d[k] for k in entries})
            T = tf.data.Dataset.from_tensor_slices(
                np.array(data['1_T_2'], dtype=np.float32))
            if config['preprocessing']['upright']:
                T = tf.data.Dataset.zip(
                        (T,
                         tf.data.Dataset.from_tensor_slices(data['rot']),
                         tf.data.Dataset.from_tensor_slices(data['rot2'])))
                T = T.map_parallel(_rotate_extrinsics)
            dataset = tf.data.Dataset.zip(
                (dataset, dataset2, T)).map(
                    lambda d1, d2, t: {**d1, **d2, '1_T_2': t})

        return dataset
