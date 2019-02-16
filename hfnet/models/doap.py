""" Implementation of Local Descriptors Optimized for Average Precision (DOAP)
    by Kun He, Yan Lu, and Stan Sclaroff.
    Paper: https://arxiv.org/abs/1804.05312
    Website: http://cs-people.bu.edu/hekun/papers/DOAP/index.html
"""

import logging
import scipy.io as scio
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow import layers as tfl

from .base_model import BaseModel
from hfnet.settings import DATA_PATH


def normalize(image):
    image = image - tf.reduce_mean(image, axis=(1, 2, 3), keepdims=True)
    var = tf.reduce_mean(tf.square(image), axis=(1, 2, 3), keepdims=True)
    image = image / tf.sqrt(var)
    return image


class Doap(BaseModel):
    input_spec = {
            'image': {'shape': [None, None, None, 1], 'type': tf.float32},
    }
    required_config_keys = []
    default_config = {
            'use_transformer': False,
            'descriptor_layer': None,
            'weights': '',
    }

    def _model(self, inputs, mode, **config):
        assert config['weights']
        is_simplenn = 'ST' not in Path(config['weights']).stem
        path = Path(DATA_PATH, 'weights', config['weights']).as_posix()
        mat = scio.loadmat(path, struct_as_record=False, squeeze_me=True)
        net = mat['net']
        layers = net.layers

        if not is_simplenn:
            params = {p.name: p.value for p in net.params}

        tf_layers = {}
        image = normalize(inputs['image'])
        if is_simplenn or config['use_transformer']:
            previous = 'input'
            start = 0
        else:
            previous = 'xST'
            start = [l.name for l in layers].index('conv1')
        tf_layers[previous] = image
        end = len(layers) - 2  # skip loss

        for i in range(start, end+1):
            layer = layers[i]
            if is_simplenn:
                layer.inputs = previous
                previous = layer.outputs = layer.name
            logging.info(f'Reading layer {layer.name:^15} '
                         f'{layer.inputs:} --> {layer.outputs}')

            if layer.type in ['dagnn.Conv', 'conv']:
                if is_simplenn:
                    w, b = layer.weights  # weight, bias
                    if len(w.shape) == 3:  # Matlab skips channels if 1
                        w = w[:, :, np.newaxis]
                    k1, k2, c_in, c_out = w.shape
                else:
                    shape = layer.block.size
                    k1, k2, c_in, c_out = shape
                    w, b = layer.params  # weight, bias
                    w, b = params[w], params[b]
                    w = w.reshape(shape)
                assert c_out == b.shape[0]

                if is_simplenn:
                    stride = np.array([layer.stride]*2)
                    pad = np.array([layer.pad]*4).reshape((2, 2))
                    dilate = np.array([layer.dilate]*2)
                else:
                    stride = layer.block.stride
                    pad = layer.block.pad.reshape((2, 2))
                    dilate = layer.block.dilate
                assert np.all(dilate == 1)
                conv_pad = 'valid'

                if layer.name == 'conv7':  # preserve shape in last conv
                    pad = np.zeros_like(pad)
                    conv_pad = 'same'

                pad = np.concatenate([[[0, 0]], pad, [[0, 0]]])
                input_tensor = tf_layers[layer.inputs]
                input_tensor = tf.pad(input_tensor, pad)

                w = tf.constant_initializer(w)
                b = tf.constant_initializer(b)
                tf_layer = tfl.conv2d(
                    input_tensor, c_out, (k1, k2), strides=stride,
                    padding=conv_pad, activation=None,
                    kernel_initializer=w, bias_initializer=b)
                tf_layers[layer.outputs] = tf_layer

                logging.info(f'Output {str(tf_layer.shape):>10} {pad.max()}')

            elif layer.type in ['dagnn.ReLU', 'relu']:
                assert (layer.leak if is_simplenn else layer.block.leak) == 0
                input_tensor = tf_layers[layer.inputs]
                tf_layer = tf.nn.relu(input_tensor)
                tf_layers[layer.outputs] = tf_layer

            elif layer.type in ['dagnn.BatchNorm', 'bnorm']:
                if is_simplenn:
                    w, b, m = layer.weights  # weight, bias, moments
                    epsilon = layer.epsilon
                else:
                    w, b, m = layer.params  # weight, bias, moments
                    w, b, m = params[w], params[b], params[m]
                    epsilon = layer.block.epsilon
                mean, std = m[:, 0], m[:, 1]

                input_tensor = tf_layers[layer.inputs]
                w = tf.constant_initializer(w)
                b = tf.constant_initializer(b)
                mean = tf.constant_initializer(mean)
                var = tf.constant_initializer(std**2)
                tf_layer = tfl.batch_normalization(
                    input_tensor, epsilon=epsilon,
                    beta_initializer=b, gamma_initializer=w,
                    moving_mean_initializer=mean,
                    moving_variance_initializer=var,
                    training=False)
                tf_layers[layer.outputs] = tf_layer

            elif layer.type in ['dagnn.DropOut', 'dropout']:
                input_tensor = tf_layers[layer.inputs]
                tf_layers[layer.outputs] = input_tensor

            elif layer.type == 'dagnn.AffineGridGenerator':
                raise NotImplementedError

            elif layer.type == 'models.PaddedBilinear':
                raise NotImplementedError

            else:
                raise NotImplementedError(layer.type)

        if config['descriptor_layer'] is None:
            config['descriptor_layer'] = previous if is_simplenn else 'logits'

        desc = tf_layers[config['descriptor_layer']]
        desc = tf.nn.l2_normalize(desc, axis=-1)
        return {'local_descriptor_map': desc,
                'input_shape': tf.shape(image)[tf.newaxis, 1:]}

    def _loss(self, outputs, inputs, **config):
        raise NotImplementedError

    def _metrics(self, outputs, inputs, **config):
        raise NotImplementedError

    def load(self, checkpoint_path, flexible_restore=True):
        return
