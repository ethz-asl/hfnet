import tensorflow as tf
from tensorflow.contrib import slim


def image_normalization(image, pixel_value_offset=128.0, pixel_value_scale=128.0):
    return tf.div(tf.subtract(image, pixel_value_offset), pixel_value_scale)


def delf_attention(feature_map, config, is_training, arg_scope=None):
    with tf.variable_scope('attonly/attention/compute'):
        with slim.arg_scope(arg_scope):
            is_training = config['train_attention'] and is_training
            with slim.arg_scope([slim.conv2d, slim.batch_norm],
                                trainable=is_training):
                with slim.arg_scope([slim.batch_norm], is_training=is_training):
                    attention = slim.conv2d(
                            feature_map, 512, config['attention_kernel'], rate=1,
                            activation_fn=tf.nn.relu, scope='conv1')
                    attention = slim.conv2d(
                            attention, 1, config['attention_kernel'], rate=1,
                            activation_fn=None, normalizer_fn=None, scope='conv2')
                    attention = tf.nn.softplus(attention)
    if config['normalize_feature_map']:
        feature_map = tf.nn.l2_normalize(feature_map, -1)
    descriptor = tf.reduce_sum(feature_map*attention, axis=[1, 2])
    if config['normalize_average']:
        descriptor /= tf.reduce_sum(attention, axis=[1, 2])
    return descriptor


def vlad(feature_map, config, training, mask=None):
    with tf.variable_scope('vlad'):
        if config['intermediate_proj']:
            with slim.arg_scope([slim.conv2d, slim.batch_norm], trainable=training):
                with slim.arg_scope([slim.batch_norm], is_training=training):
                    feature_map = slim.conv2d(
                            feature_map, config['intermediate_proj'], 1, rate=1,
                            activation_fn=None, normalizer_fn=slim.batch_norm,
                            weights_initializer=slim.initializers.xavier_initializer(),
                            trainable=training, scope='pre_proj')

        batch_size = tf.shape(feature_map)[0]
        feature_dim = feature_map.shape[-1]

        with slim.arg_scope([slim.batch_norm], trainable=training, is_training=training):
            memberships = slim.conv2d(
                    feature_map, config['n_clusters'], 1, rate=1,
                    activation_fn=None, normalizer_fn=slim.batch_norm,
                    weights_initializer=slim.initializers.xavier_initializer(),
                    trainable=training, scope='memberships')
            memberships = tf.nn.softmax(memberships, axis=-1)

        clusters = slim.model_variable(
                'clusters', shape=[1, 1, 1, config['n_clusters'], feature_dim],
                initializer=slim.initializers.xavier_initializer(), trainable=training)
        residuals = clusters - tf.expand_dims(feature_map, axis=3)
        residuals *= tf.expand_dims(memberships, axis=-1)
        if mask is not None:
            residuals *= tf.to_float(mask)[..., tf.newaxis, tf.newaxis]
        descriptor = tf.reduce_sum(residuals, axis=[1, 2])

        descriptor = tf.nn.l2_normalize(descriptor, axis=1)  # intra-normalization
        descriptor = tf.reshape(descriptor,
                                [batch_size, feature_dim*config['n_clusters']])
        descriptor = tf.nn.l2_normalize(descriptor, axis=1)
        return descriptor


def dimensionality_reduction(descriptor, config):
    descriptor = tf.nn.l2_normalize(descriptor, -1)
    reg = slim.l2_regularizer(config['proj_regularizer']) \
        if config['proj_regularizer'] else None
    descriptor = slim.fully_connected(
            descriptor,
            config['dimensionality_reduction'],
            activation_fn=None,
            weights_initializer=slim.initializers.xavier_initializer(),
            trainable=True,
            weights_regularizer=reg,
            scope='dimensionality_reduction')
    descriptor = tf.nn.l2_normalize(descriptor, -1)
    return descriptor


def triplet_loss(outputs, inputs, **config):
    distance_p = tf.norm(outputs['descriptor_image'] - outputs['descriptor_p'], axis=1)
    distance_n = tf.norm(outputs['descriptor_image'] - outputs['descriptor_n'], axis=1)
    if config['loss_in']:
        loss = tf.maximum(distance_p + config['triplet_margin'] - distance_n, 0)
        if config['loss_squared']:
            loss = tf.square(loss)
    else:
        dp = tf.square(distance_p) if config['loss_squared'] else distance_p
        dn = tf.square(distance_n) if config['loss_squared'] else distance_n
        loss = dp + tf.maximum(config['triplet_margin'] - dn, 0)
    return [tf.reduce_mean(i) for i in [loss, distance_p, distance_n]]


def vlad_legacy(inputs, num_clusters, assign_weight_initializer=None,
                cluster_initializer=None, skip_postnorm=False):
    """Implementation from https://github.com/uzh-rpg/netvlad_tf_open
    """
    K = num_clusters
    D = inputs.get_shape()[-1]

    # soft-assignment.
    s = tf.layers.conv2d(inputs, K, 1, use_bias=False,
                         kernel_initializer=assign_weight_initializer,
                         name='assignment')
    a = tf.nn.softmax(s)

    # Dims used hereafter: batch, H, W, desc_coeff, cluster
    # Move cluster assignment to corresponding dimension.
    a = tf.expand_dims(a, -2)

    # VLAD core.
    C = tf.get_variable('cluster_centers', [1, 1, 1, D, K],
                        initializer=cluster_initializer,
                        dtype=inputs.dtype)

    v = tf.expand_dims(inputs, -1) + C
    v = a * v
    v = tf.reduce_sum(v, axis=[1, 2])
    v = tf.transpose(v, perm=[0, 2, 1])

    if not skip_postnorm:
        # Result seems to be very sensitive to the normalization method
        # details, so sticking to matconvnet-style normalization here.
        v = matconvnetNormalize(v, 1e-12)
        v = tf.transpose(v, perm=[0, 2, 1])
        v = matconvnetNormalize(tf.layers.flatten(v), 1e-12)

    return v


def matconvnetNormalize(inputs, epsilon):
    """Implementation from https://github.com/uzh-rpg/netvlad_tf_open
    """
    return inputs / tf.sqrt(tf.reduce_sum(inputs ** 2, axis=-1, keep_dims=True)
                            + epsilon)
