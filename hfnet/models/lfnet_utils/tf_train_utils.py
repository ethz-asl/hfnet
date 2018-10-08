# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

def get_optimizer(method, global_step, learning_rate, loss, var_list, max_grad_norm=None, show_var_and_grad=False, verbose=True):
    method = method.lower()
    if method == 'adam':
        optim = tf.train.AdamOptimizer(learning_rate)
    elif method == 'momentum':
        optim = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
    elif method == 'ftrl':
        optim = tf.train.FtrlOptimizer(learning_rate)
    elif method == 'rmsprop':
        optim = tf.train.RMSPropOptimizer(learning_rate)
    else:
        raise Exception('Invalid optimizer method: {}'.format(method))

    if verbose:
        print('========== get_optimizer ({}) =========='.format(method))
        print(optim)

    with tf.variable_scope('Optimization') as sc:
        # gradient clipping
        if max_grad_norm is not None:
            grads_and_vars = optim.compute_gradients(loss, var_list)
            new_grads_and_vars = []
            for idx, (grad, var) in enumerate(grads_and_vars):
                if grad is not None and var in var_list:
                    new_grads_and_vars.append(
                        (tf.clip_by_norm(grad, max_grad_norm), var))
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            for g, v in grads_and_vars:
                if verbose:
                    print(v.name, v.shape)
                if show_var_and_grad:
                    tf.summary.histogram(v.name, v)
                    tf.summary.histogram(v.name+'/gradient', g)

            # tf Batch norm requires update_ops to be added as a train_op dependency.
            with tf.control_dependencies(update_ops):
                minimize_op = optim.apply_gradients(
                    new_grads_and_vars, global_step=global_step)
        else:
            grads_and_vars = optim.compute_gradients(loss, var_list)

            for g, v in grads_and_vars:
                if verbose:
                    print(v.name, v.shape)
                if show_var_and_grad:
                    tf.summary.histogram(v.name, v)
                    tf.summary.histogram(v.name+'/gradient', g)

            # tf Batch norm requires update_ops to be added as a train_op dependency.
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                minimize_op = optim.minimize(
                    loss, var_list=var_list, global_step=global_step)

        if verbose:
            print('=======================================')

        return minimize_op

def get_custom_optimizer(method, global_step, learning_rate, loss, var_list, max_grad_norm=None, check_numerics=False, verbose=True, show_summary=False):

    method = method.lower()
    if method == 'adam':
        optim = tf.train.AdamOptimizer(learning_rate)
    elif method == 'momentum':
        optim = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
    elif method == 'ftrl':
        optim = tf.train.FtrlOptimizer(learning_rate)
    elif method == 'rmsprop':
        optim = tf.train.RMSPropOptimizer(learning_rate)
    else:
        raise Exception('Invalid optimizer method: {}'.format(method))

    if verbose:
        print('========== get_optimizer ({}) =========='.format(method))
        print(optim)

    with tf.variable_scope('Optimization') as sc:

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            grads_and_vars = optim.compute_gradients(loss, var_list=var_list)

            if max_grad_norm is not None:
                new_grads_and_vars = []
                # check whether gradients contain large value (then clip), NaN and InF
                for idx, (grad, var) in enumerate(grads_and_vars):
                    if grad is not None and var in var_list:
                        grad = tf.clip_by_norm(grad, max_grad_norm)
                        new_grads_and_vars.append((grad, var))
                grads_and_vars = new_grads_and_vars

            # Check numerics and report if something is going on. This will make the backward pass stop and skip the batch
            if check_numerics:
                new_grads_and_vars = []
                for idx, (grad, var) in enumerate(grads_and_vars):
                    if grad is not None and var in var_list:
                        grad = tf.check_numerics(
                            grad, "Numerical error in gradient for {}".format(
                                var.name))
                        new_grads_and_vars.append((grad, var))
                grads_and_vars = new_grads_and_vars

            # Summarize all gradients
            for grad, var in grads_and_vars:
                if verbose:
                    print(var.name, var.shape)
                if show_summary:
                    tf.summary.histogram(var.name, var)
                    tf.summary.histogram(var.name+'/gradient', grad)

            minimize_op = optim.apply_gradients(grads_and_vars, global_step=global_step)

    if verbose:
        print('=======================================')

    return minimize_op

def get_piecewise_lr(global_step, boundaries, lr_values, show_summary=True):
    ''' Piesewise learning rate'''
    # args example: (from https://github.com/tensorflow/models/blob/master/official/resnet/imagenet_main.py)
    # initial_learning_rate = 0.1 * params['batch_size'] / 128
    # batches_per_epoch = _NUM_IMAGES['train'] / params['batch_size']
    # boundaries = [int(batches_per_epoch * epoch) for epoch in [100, 150, 200]]
    # lr_values = [initial_learning_rate * decay for decay in [1, 0.1, 0.01, 0.001]]
    learning_rate = tf.train.piecewise_constant(
        tf.cast(global_step, tf.int32), boundaries, lr_values)
    learning_rate = tf.identity(learning_rate, name='learning_rate')
    if show_summary:
        tf.summary.scalar('learning_rate', learning_rate)
    return learning_rate

def get_activation_fn(act_type='relu', **kwargs):
    act_type = act_type.lower()
    act_fn = None
    if act_type == 'relu':
        act_fn = tf.nn.relu
    elif act_type == 'leaky_relu':
        alpha = kwargs.pop('alpha', 0.2)
        act_fn = lambda x, name=None : tf.nn.leaky_relu(x, alpha, name=name)
    elif act_type == 'sigmoid':
        act_fn = tf.nn.sigmoid
    elif act_type == 'tanh':
        act_fn = tf.nn.tanh
    elif act_type == 'crelu':
        act_fn = tf.nn.crelu
    elif act_type == 'elu':
        act_fn = tf.nn.elu

    print('Act-Fn: ', act_fn)
    return act_fn
