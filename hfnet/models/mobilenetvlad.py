import tensorflow as tf
from tensorflow.contrib import slim

from .base_model import BaseModel, Mode
from .backbones import mobilenet_v2 as mobilenet
from .utils.layers import vlad, dimensionality_reduction, image_normalization


class Mobilenetvlad(BaseModel):
    input_spec = {
            'image': {'shape': [None, None, None, 1], 'type': tf.float32},
    }
    required_config_keys = []
    default_config = {
            'depth_multiplier': 1.0,
            'tile_image_if_grayscale': False,
            'dropout_keep_prob': None,
            'encoder_endpoint': 'layer_18',
            'intermediate_proj': None,
            'n_clusters': 64,
            'dimensionality_reduction': None,
            'proj_regularizer': 0.,
            'train_backbone': True,
            'train_vlad': True,
            'local_descriptor_layer': None,
    }

    def _model(self, inputs, mode, **config):
        image = image_normalization(inputs['image'])
        if config['tile_image_if_grayscale']:
            if image.shape[-1] == 1:
                image = tf.tile(image, [1, 1, 1, 3])

        is_training = config['train_backbone'] and (mode == Mode.TRAIN)
        with slim.arg_scope(mobilenet.training_scope(
                is_training=is_training, dropout_keep_prob=config['dropout_keep_prob'])):
            _, encoder = mobilenet.mobilenet(image, num_classes=None, base_only=True,
                                             depth_multiplier=config['depth_multiplier'],
                                             final_endpoint=config['encoder_endpoint'])
        feature_map = encoder[config['encoder_endpoint']]
        descriptor = vlad(feature_map, config, mode == Mode.TRAIN)
        if config['dimensionality_reduction']:
            descriptor = dimensionality_reduction(descriptor, config)

        ret = {'global_descriptor': descriptor}
        if config['local_descriptor_layer']:
            desc = encoder[config['local_descriptor_layer']]
            ret['local_descriptor_map'] = tf.nn.l2_normalize(desc, axis=-1)
        return ret

    def _descriptor_l2_error(self, inputs, outputs):
        dist = tf.square(inputs['global_descriptor']
                         - outputs['global_descriptor'])
        return tf.reduce_sum(dist, axis=-1)/2

    def _loss(self, outputs, inputs, **config):
        return tf.reduce_mean(self._descriptor_l2_error(inputs, outputs))

    def _metrics(self, outputs, inputs, **config):
        return {'l2_error': self._descriptor_l2_error(inputs, outputs)}
