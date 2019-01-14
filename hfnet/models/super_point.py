import tensorflow as tf
from tensorflow import layers as tfl

from .base_model import BaseModel, Mode
from .utils.layers import simple_nms


def vgg_block(inputs, filters, kernel_size, name, data_format, training=False,
              batch_normalization=True, kernel_reg=0., **params):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        x = tfl.conv2d(inputs, filters, kernel_size, name='conv',
                       kernel_regularizer=tf.contrib.layers.l2_regularizer(
                           kernel_reg),
                       data_format=data_format, **params)
        if batch_normalization:
            x = tfl.batch_normalization(
                    x, training=training, name='bn', fused=True,
                    axis=1 if data_format == 'channels_first' else -1)
    return x


def vgg_backbone(inputs, **config):
    params_conv = {'padding': 'SAME', 'data_format': config['data_format'],
                   'activation': tf.nn.relu, 'batch_normalization': False,
                   'training': config['training'],
                   'kernel_reg': config.get('kernel_reg', 0.)}
    params_pool = {'padding': 'SAME', 'data_format': config['data_format']}

    with tf.variable_scope('vgg', reuse=tf.AUTO_REUSE):
        x = vgg_block(inputs, 64, 3, 'conv1_1', **params_conv)
        x = vgg_block(x, 64, 3, 'conv1_2', **params_conv)
        x = tfl.max_pooling2d(x, 2, 2, name='pool1', **params_pool)

        x = vgg_block(x, 64, 3, 'conv2_1', **params_conv)
        x = vgg_block(x, 64, 3, 'conv2_2', **params_conv)
        x = tfl.max_pooling2d(x, 2, 2, name='pool2', **params_pool)

        x = vgg_block(x, 128, 3, 'conv3_1', **params_conv)
        x = vgg_block(x, 128, 3, 'conv3_2', **params_conv)
        x = tfl.max_pooling2d(x, 2, 2, name='pool3', **params_pool)

        x = vgg_block(x, 128, 3, 'conv4_1', **params_conv)
        x = vgg_block(x, 128, 3, 'conv4_2', **params_conv)
    return x


def detector_head(inputs, **config):
    params_conv = {'padding': 'SAME', 'data_format': config['data_format'],
                   'batch_normalization': False,
                   'training': config['training'],
                   'kernel_reg': config.get('kernel_reg', 0.)}
    cfirst = config['data_format'] == 'channels_first'
    cindex = 1 if cfirst else -1  # index of the channel

    with tf.variable_scope('detector', reuse=tf.AUTO_REUSE):
        x = vgg_block(inputs, 256, 3, 'conv1',
                      **{'activation': tf.nn.relu, **params_conv})
        x = vgg_block(x, 1+pow(config['grid_size'], 2), 1, 'conv2',
                      **{'activation': None, **params_conv})

        prob = tf.nn.softmax(x, axis=cindex)
        # Strip the extra “no interest point” dustbin
        prob = prob[:, :-1, :, :] if cfirst else prob[:, :, :, :-1]
        prob = tf.depth_to_space(
                prob, config['grid_size'],
                data_format='NCHW' if cfirst else 'NHWC')
        prob = tf.squeeze(prob, axis=cindex)

    return {'logits': x, 'scores': prob}


def descriptor_head(inputs, **config):
    params_conv = {'padding': 'SAME', 'data_format': config['data_format'],
                   'batch_normalization': False,
                   'training': config['training'],
                   'kernel_reg': config.get('kernel_reg', 0.)}
    cfirst = config['data_format'] == 'channels_first'
    cindex = 1 if cfirst else -1  # index of the channel

    with tf.variable_scope('descriptor', reuse=tf.AUTO_REUSE):
        x = vgg_block(inputs, 256, 3, 'conv1',
                      **{'activation': tf.nn.relu, **params_conv})
        x = vgg_block(x, config['descriptor_size'], 1, 'conv2',
                      **{'activation': None, **params_conv})
        desc = tf.nn.l2_normalize(x, cindex)

    return {'local_descriptor_map': desc}


class SuperPoint(BaseModel):
    input_spec = {
            'image': {'shape': [None, None, None, 1], 'type': tf.float32}
    }
    required_config_keys = []
    default_config = {
            'data_format': 'channels_last',
            'descriptor_size': 256,
            'grid_size': 8,
            'detector_threshold': 0.015,
            'nms_radius': 0,
            'num_keypoints': 0,
    }

    def _model(self, inputs, mode, **config):
        assert mode != Mode.TRAIN
        config['training'] = False

        image = inputs['image'] / 255.  # normalize in [0, 1]
        if config['data_format'] == 'channels_first':
            image = tf.transpose(image, [0, 3, 1, 2])

        size = tf.shape(image)[1:3]
        target_size = tf.to_int64(tf.floor(tf.to_float(size)/8)*8)
        image = image[:, :target_size[0], :target_size[1]]

        features = vgg_backbone(image, **config)
        detections = detector_head(features, **config)
        descriptors = descriptor_head(features, **config)
        results = {**detections, **descriptors}

        if mode == Mode.PRED:
            # Batch size 1 required
            scores = results['scores_dense']
            if config['local']['nms']:
                scores = simple_nms(scores, config['local']['nms'])
            with tf.name_scope('keypoint_extraction'):
                keypoints = tf.where(tf.greater_equal(
                    scores[0], config['local']['detector_threshold']))
                scores = tf.gather_nd(scores[0], keypoints)
            if config['local']['num_keypoints']:
                with tf.name_scope('top_k_keypoints'):
                    k = tf.constant(config['local']['num_keypoints'], name='k')
                    k = tf.minimum(tf.shape(scores)[0], k)
                    scores, indices = tf.nn.top_k(scores, k)
                    keypoints = tf.to_int32(tf.gather(
                        tf.to_float(keypoints), indices))
            keypoints, scores = keypoints[None], scores[None]
            keypoints = keypoints[..., ::-1]  # x-y convention
            with tf.name_scope('descriptor_sampling'):
                desc = results['local_descriptor_map']
                scaling = ((tf.to_float(tf.shape(desc)[1:3]) - 1.)
                           / (tf.to_float(tf.shape(image)[1:3]) - 1.))
                local_descriptors = tf.contrib.resampler.resampler(
                    desc, scaling*tf.to_float(keypoints))
                local_descriptors = tf.nn.l2_normalize(local_descriptors, -1)
            results = {**results, 'keypoints': keypoints, 'scores': scores,
                       'local_descriptors': local_descriptors}

        return results

    def _loss(self, outputs, inputs, **config):
        raise NotImplementedError

    def _metrics(self, outputs, inputs, **config):
        raise NotImplementedError
