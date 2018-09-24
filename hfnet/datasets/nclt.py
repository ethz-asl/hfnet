import numpy as np
import tensorflow as tf
import cv2
import re
from pathlib import Path

from .base_dataset import BaseDataset
from hfnet.settings import DATA_PATH


class Nclt(BaseDataset):
    default_config = {
        'validation_size': 200,
        'cache_in_memory': False,
        'camera': 4,
        'preprocessing': {
            'undistort': False,
            'grayscale': True,
            'resize': [640, 488],
        }
    }
    dataset_folder = 'nclt'

    def _init_dataset(self, **config):
        base_path = Path(DATA_PATH, self.dataset_folder)
        self.split_names = []
        paths = {}

        # Triplets for training
        if 'training_triplets' in config:
            self.split_names.extend(['training', 'validation'])
            training_triplets = np.load(Path(base_path, config['training_triplets']))
            if 'validation_triplets' in config:
                validation_triplets = np.load(
                        Path(base_path, config['validation_triplets']))
                validation_triplets = validation_triplets[:config['validation_size']]
            else:
                validation_triplets = training_triplets[:config['validation_size']]
                training_triplets = training_triplets[config['validation_size']:]
            splits = {'validation': validation_triplets, 'training': training_triplets}
            for s in splits:
                paths[s] = {'image': [], 'p': [], 'n': []}
                for triplet in splits[s]:
                    for (seq, time), e in zip(triplet, ['image', 'p', 'n']):
                        paths[s][e].append(
                                str(Path(base_path, '{}/lb3/Cam{}/{}.tiff'.format(
                                    seq, config['camera'], time))))
        # Images for testing
        if 'test_sequences' in config:
            self.split_names.append('test')
            seqs = config['test_sequences']
            if not isinstance(seqs, list):
                seqs = [seqs]
            paths['test'] = {'name': [], 'image': []}
            for seq in seqs:
                seq_poses = self.get_pose_file(seq)
                timestamps = seq_poses['time'].astype(str).tolist()
                paths['test']['name'].extend([seq+'/'+t for t in timestamps])
                paths['test']['image'].extend(
                        [str(Path(base_path, '{}/lb3/Cam{}/{}.tiff'.format(
                            seq, config['camera'], n))) for n in timestamps])
        return paths

    @staticmethod
    def get_pose_file(sequence):
        return np.loadtxt(
                Path(DATA_PATH, Nclt.dataset_folder, 'pose_{}.csv'.format(sequence)),
                dtype={'names': ('time', 'x', 'y', 'angle'),
                       'formats': (np.int, np.float, np.float, np.float)},
                delimiter=',', skiprows=1)

    class Undistort(object):
        def __init__(self, fin, scale=1.0, fmask=None):
            self.fin = fin
            # read in distort
            with open(fin, 'r') as f:
                header = f.readline().rstrip()
                chunks = re.sub(r'[^0-9,]', '', header).split(',')
                self.mapu = np.zeros((int(chunks[1]), int(chunks[0])),
                                     dtype=np.float32)
                self.mapv = np.zeros((int(chunks[1]), int(chunks[0])),
                                     dtype=np.float32)
                for line in f.readlines():
                    chunks = line.rstrip().split(' ')
                    self.mapu[int(chunks[0]), int(chunks[1])] = float(chunks[3])
                    self.mapv[int(chunks[0]), int(chunks[1])] = float(chunks[2])
            # generate a mask
            self.mask = np.ones(self.mapu.shape, dtype=np.uint8)
            self.mask = cv2.remap(self.mask, self.mapu, self.mapv, cv2.INTER_LINEAR)
            kernel = np.ones((30, 30), np.uint8)
            self.mask = cv2.erode(self.mask, kernel, iterations=1)
            # crop black regions out
            h, w = self.mask.shape
            self.x_lim = [f(np.where(self.mask[int(h/2), :])[0])
                          for f in [np.min, np.max]]
            self.y_lim = [f(np.where(self.mask[:, int(w/2)])[0])
                          for f in [np.min, np.max]]

        def undistort(self, img, crop=True):
            undistorted = cv2.resize(cv2.remap(img, self.mapu, self.mapv,
                                               cv2.INTER_LINEAR),
                                     (self.mask.shape[1], self.mask.shape[0]),
                                     interpolation=cv2.INTER_CUBIC)
            if crop:
                undistorted = undistorted[self.y_lim[0]:self.y_lim[1],
                                          self.x_lim[0]:self.x_lim[1]]
            return undistorted

    def _get_data(self, paths, split_name, **config):
        def _read_image(path):
            return cv2.imread(path.decode('utf-8'))

        def _undistort():
            undistort_file = Path(DATA_PATH, self.dataset_folder, 'undistort_maps/',
                                  'U2D_Cam{}_1616X1232.txt'.format(config['camera']))
            undistort_map = self.Undistort(undistort_file)
            return undistort_map.undistort

        def _preprocess(image):
            if config['preprocessing']['undistort']:
                image = tf.py_func(_undistort(), [image], tf.uint8)
                image.set_shape([None, None, 3])
            image = tf.image.rot90(image, k=3)
            if config['preprocessing']['grayscale']:
                image = tf.image.rgb_to_grayscale(image)
            if config['preprocessing']['resize']:
                image = tf.image.resize_images(image, config['preprocessing']['resize'],
                                               method=tf.image.ResizeMethod.BILINEAR)
            return image

        datasets = {}
        for e in paths[split_name]:
            d = tf.data.Dataset.from_tensor_slices(paths[split_name][e])
            if e != 'name':
                d = d.map(lambda path: tf.py_func(_read_image, [path], tf.uint8),
                          num_parallel_calls=13)
                d = d.map(_preprocess,
                          num_parallel_calls=13)
            datasets[e] = d
        dataset = tf.data.Dataset.zip(datasets)

        if config['cache_in_memory']:
            tf.logging.info('Caching dataset, fist access will take some time.')
            dataset = dataset.cache()

        return dataset
