import tensorflow as tf
import numpy as np
from argparse import Namespace
from pathlib import Path
import pickle

from .base_model import BaseModel, Mode
from hfnet.settings import DATA_PATH
from hfnet.utils import tools

from .lfnet_utils.det_tools import (instance_normalization, get_degree_maps)
from .lfnet_utils.inference import (
    build_multi_scale_deep_detector_3DNMS, build_multi_scale_deep_detector,
    build_patch_extraction)
from .lfnet_utils.tf_train_utils import get_activation_fn
from .lfnet_utils.tf_layer_utils import (
    batch_norm_act, conv2d_fixed_padding, conv2d, fully_connected,
    conv2d_custom)


def building_block(
        inputs, out_channels,
        projection_shortcut,
        stride,
        scope,
        conv_ksize=3,
        use_xavier=True,
        activation_fn=tf.nn.relu,
        perform_bn=False,
        bn_decay=None,
        bn_affine=True,
        is_training=None,
        use_bias=True):

    with tf.variable_scope(scope):
        curr_in = inputs
        shortcut = curr_in # activate_before_residual=False
        curr_in = batch_norm_act(curr_in, activation_fn=activation_fn,
                                 perform_bn=perform_bn,
                                 is_training=is_training,
                                 bn_decay=bn_decay,
                                 bn_affine=bn_affine,
                                 bnname='pre-bn'
                                )
        # The projection shortcut should come after the first batch norm
        # and ReLU since it performs a 1x1 convolution.
        if projection_shortcut is not None:
            shortcut = projection_shortcut(curr_in) # discard previous shortcut

        # conv-bn-act
        curr_in = conv2d_fixed_padding(curr_in, out_channels,
                         kernel_size=conv_ksize,
                         scope='conv1',
                         stride=stride,
                         use_xavier=use_xavier,
                         use_bias=use_bias,
                        )
        curr_in = batch_norm_act(curr_in, activation_fn=activation_fn,
                                 perform_bn=perform_bn,
                                 is_training=is_training,
                                 bn_decay=bn_decay,
                                 bn_affine=bn_affine,
                                 bnname='mid-bn'
                                )
        # conv only
        curr_in = conv2d_fixed_padding(curr_in, out_channels,
                         kernel_size=conv_ksize,
                         scope='conv2',
                         stride=1,
                         use_xavier=use_xavier,
                         use_bias=use_bias,
                        )
        return curr_in + shortcut


def get_detector(inputs, is_training,
              num_block=3,
              num_channels=16,
              conv_ksize=3,
              ori_conv_ksize=None,
              activation_fn=tf.nn.relu,
              use_xavier=True, perform_bn=True,
              bn_trainable=True,
              bn_decay=None, bn_affine=True, use_bias=True,
              min_scale=2**-3, max_scale=1, num_scales=9,
              reuse=False, name='ConvOnlyResNet'):
    num_conv = 0
    if ori_conv_ksize is None:
        ori_conv_ksize = conv_ksize # same ksize as others
    with tf.variable_scope(name, reuse=reuse) as net_sc:
        curr_in = tf.identity(inputs)

        # init-conv
        curr_in = conv2d_fixed_padding(curr_in, num_channels,
                        kernel_size=conv_ksize, scope='init_conv',
                        use_xavier=use_xavier, use_bias=use_bias)
        num_conv += 1
        for i in range(num_block):
            curr_in = building_block(curr_in, num_channels, None,
                        stride=1, scope='block-{}'.format(i+1),
                        conv_ksize=conv_ksize,
                        use_xavier=use_xavier,
                        activation_fn=activation_fn,
                        perform_bn=perform_bn,
                        bn_decay=bn_decay, bn_affine=bn_affine,
                        is_training=is_training, use_bias=use_bias
                        )
            num_conv += 2
        curr_in = batch_norm_act(curr_in, activation_fn=activation_fn,
                                 perform_bn=perform_bn,
                                 is_training=is_training,
                                 bn_decay=bn_decay,
                                 bn_affine=bn_affine,
                                 bnname='fin-bn')

        feat_maps = tf.identity(curr_in)

        if num_scales == 1:
            scale_factors = [1.0]
        else:
            scale_log_factors = np.linspace(
                np.log(max_scale), np.log(min_scale), num_scales)
            scale_factors = np.exp(scale_log_factors)
        print('Scales ({:2f}~{:.2f} #{}): {}'.format(
            min_scale, max_scale, num_scales, scale_factors))
        score_maps_list = []

        base_height_f = tf.to_float(tf.shape(curr_in)[1])
        base_width_f = tf.to_float(tf.shape(curr_in)[2])

        for i, s in enumerate(scale_factors):
            # scale are defined by extracted patch size
            # (s of s*default_patch_size) so we need use inv-scale
            # for resizing images
            inv_s = 1.0 / s
            feat_height = tf.cast(base_height_f * inv_s+0.5, tf.int32)
            feat_width = tf.cast(base_width_f * inv_s+0.5, tf.int32)
            rs_feat_maps = tf.image.resize_images(
                curr_in, tf.stack([feat_height, feat_width]))
            score_maps = conv2d_fixed_padding(
                    rs_feat_maps, 1,
                    kernel_size=conv_ksize, scope='score_conv_{}'.format(i),
                    use_xavier=use_xavier, use_bias=use_bias)
            score_maps_list.append(score_maps)

        num_conv += 1

        # orientation (initial map start from 0.0)
        ori_W_init = tf.zeros_initializer
        # init with 1 for cos(q), 0 for sin(q)
        ori_b_init = tf.constant(np.array([1,0], dtype=np.float32))
        ori_maps = conv2d_custom(curr_in, 2,
                    kernel_size=ori_conv_ksize, scope='ori_conv',
                    W_initializer=ori_W_init,
                    b_initializer=ori_b_init)
        ori_maps = tf.nn.l2_normalize(ori_maps, dim=-1)

        all_var_list = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, net_sc.name)

        var_list = []
        mso_var_list = []

        for var in all_var_list:
            if 'ori_conv' in var.name:
                mso_var_list.append(var)
                # if not reuse:
                    # tf.summary.histogram(var.name, var)
            else:
                var_list.append(var)

        endpoints = {}
        endpoints['ori_maps'] = ori_maps
        endpoints['var_list'] = var_list
        endpoints['mso_var_list'] = mso_var_list
        endpoints['mso'] = True
        endpoints['multi_scores'] = True
        endpoints['scale_factors'] = scale_factors
        endpoints['feat_maps'] = feat_maps
        endpoints['pad_size'] = num_conv * (conv_ksize//2)
        print('PAD={}, #conv={}, ksize={} ori-ksize={}'.format(
            endpoints['pad_size'], num_conv, conv_ksize, ori_conv_ksize))
        return score_maps_list, endpoints


def get_descriptor(inputs, is_training,
              out_dim=128,
              init_num_channels=64,
              num_conv_layers=3,
              conv_ksize=3,
              activation_fn=tf.nn.relu,
              use_xavier=True, perform_bn=True,
              bn_trainable=True,
              bn_decay=None, bn_affine=True, use_bias=True,
              feat_norm='l2norm',
              reuse=False, name='SimpleDesc'):

    channels_list = [init_num_channels * 2**i for i in range(num_conv_layers)]
    print('===== {} (reuse={}) ====='.format(name, reuse))

    with tf.variable_scope(name, reuse=reuse) as net_sc:
        curr_in = inputs

        for i, num_channels in enumerate(channels_list):
            curr_in = conv2d(curr_in, num_channels,
                        kernel_size=conv_ksize, scope='conv{}'.format(i+1),
                        stride=2, padding='SAME',
                        use_xavier=use_xavier, use_bias=use_bias)
            curr_in = batch_norm_act(curr_in, activation_fn,
                                     perform_bn=perform_bn,
                                     is_training=is_training,
                                     bn_decay=bn_decay,
                                     bn_affine=bn_affine,
                                     bnname='bn{}'.format(i+1)
                                    )
            print('#{} conv-bn-act {}'.format(i+1, curr_in.shape))
        #----- FC
        curr_in = tf.layers.flatten(curr_in)
        print('FLAT {}'.format(curr_in.shape))
        curr_in = fully_connected(curr_in, 512, scope='fc1',
                                  use_xavier=use_xavier, use_bias=use_bias)
        curr_in = batch_norm_act(curr_in, activation_fn,
                                 perform_bn=perform_bn,
                                 is_training=is_training,
                                 bn_decay=bn_decay,
                                 bn_affine=bn_affine,
                                 bnname='fc-bn1'
                                )
        raw_feats = fully_connected(curr_in, out_dim, scope='fc2',
                                use_xavier=use_xavier, use_bias=use_bias)
        if feat_norm == 'l2norm':
            norm_feats = tf.nn.l2_normalize(raw_feats, dim=1)
            print('Feat-Norm: L2-NORM')
        elif feat_norm == 'inst':
            norm_feats = instance_normalization(raw_feats)
            print('Feat-Norm: INSTANCE-NORM')
        elif feat_norm == 'rootsift':
            # need pre-L2 normalization ?
            # norm_feats = tf.nn.l2_normalize(raw_feats, dim=1)
            eps = 1e-6
            norm_feats = raw_feats
            l1norm = tf.norm(norm_feats, ord=1, axis=1, keep_dims=True)
            norm_feats = norm_feats / (l1norm + eps)
            # need to avoid unstability around too small values
            norm_feats = tf.maximum(norm_feats, eps)
            norm_feats = tf.sqrt(norm_feats)
        elif feat_norm == 'rootsift2':
            # need pre-L2 normalization ?
            # because SIFT in RootSIFT has also already normalized
            eps = 1e-6
            norm_feats = tf.nn.l2_normalize(raw_feats, dim=1)
            l1norm = tf.norm(norm_feats, ord=1, axis=1, keep_dims=True)
            norm_feats = norm_feats / (l1norm + eps)
            # need to avoid unstability around too small values
            norm_feats = tf.maximum(norm_feats, eps)
            norm_feats = tf.sqrt(norm_feats)
        elif feat_norm == 'non':
          norm_feats = raw_feats
          print('Feat-Norm: Nothing')
        else:
          raise ValueError('Unknown feat_norm: {}'.format(feat_norm))
        print('FEAT {}'.format(norm_feats.shape))
        var_list = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, net_sc.name)
        endpoints = {}
        endpoints['raw_feats'] = raw_feats
        endpoints['norm_feats'] = norm_feats
        endpoints['var_list'] = var_list
        return norm_feats, endpoints


class Detector(object):
    def __init__(self, config, is_training):
        self.config = config
        self.is_training = is_training
        self.activation_fn = get_activation_fn(
            config.activ_fn, **{'alpha': config.leaky_alpha})

    def build_model(self, photos, reuse=False, name='ConvOnlyResNet'):
        conv_ksize = getattr(self.config, 'conv_ksize', 3)
        use_xavier = getattr(self.config, 'use_xavier', True)
        use_bias = getattr(self.config, 'use_bias', True)
        bn_trainable = getattr(self.config, 'bn_trainable', True)
        bn_decay = getattr(self.config, 'bn_decay', None)
        bn_affine = getattr(self.config, 'bn_affine', True)

        min_scale = self.config.net_min_scale
        max_scale = self.config.net_max_scale
        num_scales = self.config.net_num_scales

        logits, endpoints = get_detector(
                photos, self.is_training,
                num_block=self.config.net_block,
                num_channels=self.config.net_channel,
                conv_ksize=conv_ksize,
                activation_fn=self.activation_fn,
                use_xavier=use_xavier, use_bias=use_bias,
                perform_bn=self.config.perform_bn,
                bn_trainable=bn_trainable,
                bn_decay=bn_decay, bn_affine=bn_affine,
                min_scale=min_scale, max_scale=max_scale,
                num_scales=num_scales,
                reuse=reuse, name=name)
        return logits, endpoints


class Descriptor(object):
    def __init__(self, config, is_training):
        self.config = config
        self.is_training = is_training
        self.activation_fn = get_activation_fn(
            config.desc_activ_fn, **{'alpha': config.desc_leaky_alpha})

    def build_model(self, feat_maps, reuse=False, name='SimpleDesc'):
        # out_dim = getattr(self.config, 'desc_dim', 128)
        feats, endpoints = get_descriptor(
                feat_maps, self.is_training,
                out_dim=self.config.desc_dim,
                init_num_channels=self.config.desc_net_channel,
                num_conv_layers=self.config.desc_net_depth,
                conv_ksize=self.config.desc_conv_ksize,
                activation_fn=self.activation_fn,
                perform_bn=self.config.desc_perform_bn,
                feat_norm=self.config.desc_norm,
                reuse=reuse, name=name)
        return feats, endpoints


def build_networks(config, photo, is_training):
    detector = Detector(config, is_training)

    if config.input_inst_norm:
        print('Apply instance norm on input photos')
        photos1 = instance_normalization(photo)

    if config.use_nms3d:
        heatmaps, det_endpoints = build_multi_scale_deep_detector_3DNMS(
            config, detector, photo, reuse=False)
    else:
        heatmaps, det_endpoints = build_multi_scale_deep_detector(
            config, detector, photo, reuse=False)

    # extract patches
    kpts = det_endpoints['kpts']
    batch_inds = det_endpoints['batch_inds']

    kp_patches = build_patch_extraction(config, det_endpoints, photo)

    # Descriptor
    descriptor = Descriptor(config, is_training)
    desc_feats, desc_endpoints = descriptor.build_model(
        kp_patches, reuse=False) # [B*K,D]

    # scale and orientation (extra)
    scale_maps = det_endpoints['scale_maps']
    ori_maps = det_endpoints['ori_maps'] # cos/sin
    degree_maps, _ = get_degree_maps(ori_maps) # degree (rgb psuedo color code)
    kpts_scale = det_endpoints['kpts_scale']
    kpts_ori = det_endpoints['kpts_ori']
    kpts_ori = tf.atan2(kpts_ori[:,1], kpts_ori[:,0]) # radian
    kpts_scores = det_endpoints['kpts_scores']

    ops = {
        'photo': photo,
        'is_training': is_training,
        'kpts': kpts,
        'scores': kpts_scores,
        'feats': desc_feats,
        # EXTRA
        'scale_maps': scale_maps,
        'kpts_scale': kpts_scale,
        'degree_maps': degree_maps,
        'kpts_ori': kpts_ori,
    }
    return ops


class LfNet(BaseModel):
    input_spec = {
            'image': {'shape': [None, None, None, 1], 'type': tf.float32},
    }
    required_config_keys = []
    default_config = {}

    def _model(self, inputs, mode, **config):
        assert mode != Mode.TRAIN

        config_path = Path(DATA_PATH, 'weights', config['config']).as_posix()
        with open(config_path, 'rb') as f:
            original_config = pickle.load(f).__dict__
        config = tools.dict_update(original_config, config)
        namespace = Namespace(**config)

        image = inputs['image'] / 255.0
        ops = build_networks(namespace, image, False)
        ops = {k: tf.expand_dims(v, axis=0) for k, v in ops.items()}
        ret = {'keypoints': ops['kpts'],
               'scores': ops['scores'],
               'descriptors': ops['feats']}
        return ret

    def _loss(self, outputs, inputs, **config):
        raise NotImplementedError

    def _metrics(self, outputs, inputs, **config):
        raise NotImplementedError
