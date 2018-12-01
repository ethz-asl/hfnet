import tensorflow as tf
from tensorflow.contrib import slim

from .base_model import BaseModel, Mode
from .backbones import mobilenet_v2 as mobilenet
from .layers import vlad, dimensionality_reduction, image_normalization

from .backbones.utils import conv_blocks as ops
from .backbones.utils import mobilenet as lib


MOBILENET_DEF = dict(
    defaults={
        # Note: these parameters of batch norm affect the architecture
        # that's why they are here and not in training_scope.
        (slim.batch_norm,): {'center': True, 'scale': True},
        (slim.conv2d, slim.fully_connected, slim.separable_conv2d): {
            'normalizer_fn': slim.batch_norm, 'activation_fn': tf.nn.relu6
        },
        (ops.expanded_conv,): {
            'expansion_size': ops.expand_input_by_factor(6),
            'split_expansion': 1,
            'normalizer_fn': slim.batch_norm,
            'residual': True
        },
        (slim.conv2d, slim.separable_conv2d): {'padding': 'SAME'}
    },
    spec=[
        lib.op(slim.conv2d, stride=2, num_outputs=32, kernel_size=[3, 3]),
        lib.op(ops.expanded_conv,
               expansion_size=ops.expand_input_by_factor(1, divisible_by=1),
               num_outputs=16),
        lib.op(ops.expanded_conv, stride=2, num_outputs=24),
        lib.op(ops.expanded_conv, stride=1, num_outputs=24),
        lib.op(ops.expanded_conv, stride=2, num_outputs=32),
        lib.op(ops.expanded_conv, stride=1, num_outputs=64),
        lib.op(ops.expanded_conv, stride=1, num_outputs=128),  # branch here
        lib.op(ops.expanded_conv, stride=2, num_outputs=64),
        lib.op(ops.expanded_conv, stride=1, num_outputs=64),
        lib.op(ops.expanded_conv, stride=1, num_outputs=64),
        lib.op(ops.expanded_conv, stride=1, num_outputs=64),
        lib.op(ops.expanded_conv, stride=1, num_outputs=96),
        lib.op(ops.expanded_conv, stride=1, num_outputs=96),
        lib.op(ops.expanded_conv, stride=1, num_outputs=96),
        lib.op(ops.expanded_conv, stride=2, num_outputs=160),
        lib.op(ops.expanded_conv, stride=1, num_outputs=160),
        lib.op(ops.expanded_conv, stride=1, num_outputs=160),
        lib.op(ops.expanded_conv, stride=1, num_outputs=320),
        lib.op(slim.conv2d, stride=1, kernel_size=[1, 1], num_outputs=1280)
    ],
)


def local_head(features, config):
    bn_params = {
        'center': True, 'scale': True
    }
    conv_params = {
        'normalizer_fn': slim.batch_norm,
        'activation_fn': tf.nn.relu6,
        'stride': 1,
        'padding': 'SAME',
        'kernel_size': [3, 3],
    }
    last_conv_params = {
        'normalizer_fn': None,
        'activation_fn': None,
        'stride': 1,
        'padding': 'SAME',
        'kernel_size': [1, 1],
    }

    with tf.variable_scope('descriptor', reuse=tf.AUTO_REUSE):
        with slim.arg_scope([slim.conv2d], **conv_params), \
             slim.arg_scope([slim.batch_norm], **bn_params):
            desc = slim.conv2d(features, config['descriptor_dim'])
        with slim.arg_scope([slim.conv2d], **last_conv_params):
            desc = slim.conv2d(desc, config['descriptor_dim'])
        desc = tf.nn.l2_normalize(desc, -1)

    with tf.variable_scope('detector', reuse=tf.AUTO_REUSE):
        with slim.arg_scope([slim.conv2d], **conv_params), \
             slim.arg_scope([slim.batch_norm], **bn_params):
            logits = slim.conv2d(features, 128)
        with slim.arg_scope([slim.conv2d], **last_conv_params):
            logits = slim.conv2d(logits, 1+pow(config['detector_grid'], 2))
        prob_full = tf.nn.softmax(logits, axis=-1)
        prob = prob_full[:, :, :, :-1]  # Strip the “no interest point” dustbin
        prob = tf.depth_to_space(prob, config['detector_grid'])
        prob = tf.squeeze(prob, axis=-1)

    return {'local_descriptor_map': desc, 'logits': logits,
            'prob_full': prob_full, 'scores_dense': prob}


def global_head(features, training, config, mask=None):
    global_desc = vlad(features, config, training, mask=mask)
    if config['dimensionality_reduction']:
        global_desc = dimensionality_reduction(global_desc, config)
    return {'global_descriptor': global_desc}


def descriptor_global_loss(inp, out):
    d = tf.square(inp['global_descriptor'] - out['global_descriptor']),
    return tf.reduce_sum(d, axis=-1) / 2


def descriptor_local_loss(inp, out):
    d = tf.square(inp['local_descriptor_map'] - out['local_descriptor_map'])
    d = tf.reduce_sum(d, axis=-1) / 2

    mask = inp.get('local_descriptor_map_valid_mask', None)
    if mask is not None:
        mask = tf.to_float(mask)
        d = (tf.reduce_sum(d * mask, axis=[1, 2])
             / tf.reduce_sum(mask, axis=[1, 2]))
    else:
        d = tf.reduce_mean(d, axis=[1, 2])
    return d


def detector_loss(inp, out, config):
    if 'keypoint_map' in inp:  # hard labels
        labels = tf.to_float(inp['keypoint_map'][..., tf.newaxis])  # for GPU
        labels = tf.space_to_depth(labels, config['local']['detector_grid'])
        shape = tf.concat([tf.shape(labels)[:3], [1]], axis=0)
        labels = tf.argmax(tf.concat([2*labels, tf.ones(shape)], 3), axis=3)
        with tf.device('/cpu:0'):
            d = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=labels, logits=out['logits'])
        mask = None
    elif 'dense_scores' in inp:  # soft labels
        d = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=inp['dense_scores'], logits=out['logits'], dim=-1)
        mask = inp.get('dense_scores_valid_mask', None)
    else:
        raise ValueError

    if mask is not None:
        mask = tf.to_float(mask)
        d = (tf.reduce_sum(d * mask, axis=[1, 2])
             / tf.reduce_sum(mask, axis=[1, 2]))
    else:
        d = tf.reduce_mean(d, axis=[1, 2])
    return d


class HfNet(BaseModel):
    input_spec = {
            'image': {'shape': [None, None, None, 1], 'type': tf.float32},
    }
    required_config_keys = []
    default_config = {
            'depth_multiplier': 1.0,
            'dropout_keep_prob': None,
            'global_endpoint': 'layer_18',
            'local_endpoint': 'layer_7',
            'global': {
                'intermediate_proj': None,
                'n_clusters': 64,
                'dimensionality_reduction': None,
                'proj_regularizer': 0.,
            },
            'local': {
                'descriptor_dim': 256,
                'detector_grid': 8,
                'detector_threshold': 0.015,
            },
            'loss_weights': {
                'local_desc': 1,
                'global_desc': 1,
                'detector': 1,
            },
            'train_backbone': True,
            'train_vlad': True,
    }

    def _model(self, inputs, mode, **config):
        image = image_normalization(inputs['image'])
        training = (mode == Mode.TRAIN)

        size = tf.shape(image)[1:3]
        target_size = tf.to_int64(tf.floor(tf.to_float(size)/8)*8)
        image = image[:, :target_size[0], :target_size[1]]

        with slim.arg_scope(mobilenet.training_scope(
                is_training=training & config['train_backbone'],
                dropout_keep_prob=config['dropout_keep_prob'])):
            _, encoder = mobilenet.mobilenet(
                    image, num_classes=None, base_only=True,
                    depth_multiplier=config['depth_multiplier'],
                    final_endpoint=config['global_endpoint'],
                    conv_defs=MOBILENET_DEF)

        with tf.variable_scope('local_head', reuse=tf.AUTO_REUSE), \
                slim.arg_scope(mobilenet.training_scope(
                    is_training=training,
                    dropout_keep_prob=config['dropout_keep_prob'])):
            local_feat = encoder[config['local_endpoint']]
            ret_local = local_head(local_feat, config['local'])

        with tf.variable_scope('global_head', reuse=tf.AUTO_REUSE):
            global_feat = encoder[config['global_endpoint']]
            mask = inputs.get('valid_mask', None)
            if mask is not None:
                mask = tf.image.resize_nearest_neighbor(
                    tf.to_float(mask)[..., tf.newaxis],
                    tf.shape(global_feat)[1:3])[..., 0]
            ret_global = global_head(
                global_feat, training, config['global'], mask=mask)

        ret = {**ret_local, **ret_global}
        if mode == Mode.PRED:
            # Batch size 1 required
            keypoints = tf.where(tf.greater_equal(
                ret['scores_dense'][0], config['local']['detector_threshold']))
            scores = tf.gather_nd(ret['scores_dense'][0], keypoints)
            keypoints = keypoints[:, ::-1]  # x-y convention
            ret['keypoints'] = tf.expand_dims(keypoints, 0)
            ret['scores'] = tf.expand_dims(scores, 0)
        return ret

    def _loss(self, outputs, inputs, **config):
        desc_g = tf.reduce_mean(descriptor_global_loss(inputs, outputs))
        desc_l = tf.reduce_mean(descriptor_local_loss(inputs, outputs))
        detect = tf.reduce_mean(detector_loss(inputs, outputs, config))

        if config['loss_weights'] == 'uncertainties':
            init = tf.constant_initializer([1])
            logvars = [tf.get_variable(f'w{i}', (1,), initializer=init,
                                       dtype=tf.float32) for i in range(3)]
            precisions = [tf.exp(-v) for v in logvars]
            loss = desc_g*precisions[0] + logvars[0]
            loss += desc_l*precisions[1] + logvars[1]
            loss += 2*detect*precisions[2] + logvars[2]
        else:
            w = config['loss_weights']
            assert isinstance(w, dict)
            total = sum(list(w.values()))
            loss = (w['global_desc']*desc_g
                    + w['local_desc']*desc_l
                    + w['detector']*detect) / total
        return loss

    def _metrics(self, outputs, inputs, **config):
        ret = {
            'global_desc_l2': descriptor_global_loss(inputs, outputs),
            'local_desc_l2': descriptor_local_loss(inputs, outputs),
            'detector_crossentropy': detector_loss(inputs, outputs, config),
        }
        if config['loss_weights'] == 'uncertainties':
            for i in range(3):
                ret[f'logvar{i}'] = tf.get_variable(f'w{i}')
        return ret
