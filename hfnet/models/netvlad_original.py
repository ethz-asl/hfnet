import tensorflow as tf
import numpy as np

from .utils.layers import vlad_legacy
from .base_model import BaseModel


class NetvladOriginal(BaseModel):
    """Model implementation from https://github.com/uzh-rpg/netvlad_tf_open
    """
    input_spec = {
            'image': {'shape': [None, None, None, 3], 'type': tf.float32},
    }
    required_config_keys = []
    default_config = {
            'num_clusters': 64,
            'pca_dimension': 4096,
            'local_descriptor_layer': None,
    }

    def _model(self, inputs, mode, **config):
        image_batch = inputs['image']

        with tf.variable_scope('vgg16_netvlad_pca'):
            # Gray to color if necessary.
            if image_batch.shape[3] == 1:
                x = tf.nn.conv2d(image_batch, np.ones((1, 1, 1, 3)),
                                 np.ones(4).tolist(), 'VALID')
            else:
                assert image_batch.shape[3] == 3
                x = image_batch

            # Subtract trained average image.
            average_rgb = tf.get_variable('average_rgb', 3, dtype=image_batch.dtype)
            x = x - average_rgb
            endpoints = {}

            # VGG16
            def vggConv(inputs, numbers, out_dim, with_relu):
                activation = tf.nn.relu if with_relu else None
                return tf.layers.conv2d(inputs, out_dim, [3, 3], 1, padding='same',
                                        activation=activation, name='conv%s' % numbers)

            def vggPool(inputs):
                return tf.layers.max_pooling2d(inputs, 2, 2)

            x = vggConv(x, '1_1', 64, True)
            x = vggConv(x, '1_2', 64, False)
            endpoints['conv1_2'] = x
            x = vggPool(x)
            x = tf.nn.relu(x)

            x = vggConv(x, '2_1', 128, True)
            x = vggConv(x, '2_2', 128, False)
            endpoints['conv2_2'] = x
            x = vggPool(x)
            x = tf.nn.relu(x)

            x = vggConv(x, '3_1', 256, True)
            x = vggConv(x, '3_2', 256, True)
            x = vggConv(x, '3_3', 256, False)
            endpoints['conv3_3'] = x
            x = vggPool(x)
            x = tf.nn.relu(x)

            x = vggConv(x, '4_1', 512, True)
            x = vggConv(x, '4_2', 512, True)
            x = vggConv(x, '4_3', 512, False)
            endpoints['conv4_3'] = x
            x = vggPool(x)
            x = tf.nn.relu(x)

            x = vggConv(x, '5_1', 512, True)
            x = vggConv(x, '5_2', 512, True)
            x = vggConv(x, '5_3', 512, False)
            endpoints['conv5_3'] = x

            # NetVLAD
            x = tf.nn.l2_normalize(x, dim=-1)
            x = vlad_legacy(x, config['num_clusters'])

            # PCA
            x = tf.layers.conv2d(tf.expand_dims(tf.expand_dims(x, 1), 1),
                                 config['pca_dimension'], 1, 1, name='WPCA')
            x = tf.nn.l2_normalize(tf.layers.flatten(x), dim=-1)

        ret = {'global_descriptor': x}
        if config['local_descriptor_layer']:
            desc = tf.nn.l2_normalize(
                endpoints[config['local_descriptor_layer']], axis=-1)
            ret['local_descriptor_map'] = desc
        return ret

    def _loss(self, outputs, inputs, **config):
        raise NotImplementedError

    def _metrics(self, outputs, inputs, **config):
        raise NotImplementedError
