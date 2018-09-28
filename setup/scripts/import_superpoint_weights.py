import argparse
from pathlib import Path
import yaml
import torch
import tensorflow as tf

from hfnet.settings import EXPER_PATH, DATA_PATH
from hfnet.models.super_point import SuperPoint


parser = argparse.ArgumentParser()
parser.add_argument('--pytorch_weights', action='store', type=str,
                    default=str(Path(DATA_PATH, 'weights/superpoint_v1.pth')),
                    help='original Pytorch weights')
parser.add_argument('--exper_name', action='store', type=str,
                    default='super_point_original',
                    help='name of the resulting experiment')
args = parser.parse_args()
print("Converting SuperPoint weights from Pytorch to Tensorflow")

# Load weights from Torch
weights = torch.load(args.pytorch_weights)

# Initialize the Tensorflow model
config = {
        'descriptor_size': 256,
        'detection_threshold': 0.015,
        'nms': 4,
}

output_dir = Path(EXPER_PATH, args.exper_name)
output_dir.mkdir(parents=True, exist_ok=True)
with open(Path(output_dir, 'config.yml'), 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
checkpoint_path = Path(output_dir, 'model.ckpt').as_posix()

net = SuperPoint(data={}, n_gpus=1, **config)

# Assign the values to the weights
with tf.variable_scope('', reuse=tf.AUTO_REUSE):
    # Shared encoder
    with tf.variable_scope('vgg', reuse=tf.AUTO_REUSE):
        conv1_1 = tf.get_variable('conv1_1/conv/kernel')
        conv1_1 = tf.assign(conv1_1,
                            weights['conv1a.weight'].numpy().transpose([3, 2, 1, 0]))
        conv1_1_b = tf.get_variable('conv1_1/conv/bias')
        conv1_1_b = tf.assign(conv1_1_b, weights['conv1a.bias'].numpy())

        conv1_2 = tf.get_variable('conv1_2/conv/kernel')
        conv1_2 = tf.assign(conv1_2,
                            weights['conv1b.weight'].numpy().transpose([3, 2, 1, 0]))
        conv1_2_b = tf.get_variable('conv1_2/conv/bias')
        conv1_2_b = tf.assign(conv1_2_b, weights['conv1b.bias'].numpy())

        conv2_1 = tf.get_variable('conv2_1/conv/kernel')
        conv2_1 = tf.assign(conv2_1,
                            weights['conv2a.weight'].numpy().transpose([3, 2, 1, 0]))
        conv2_1_b = tf.get_variable('conv2_1/conv/bias')
        conv2_1_b = tf.assign(conv2_1_b, weights['conv2a.bias'].numpy())

        conv2_2 = tf.get_variable('conv2_2/conv/kernel')
        conv2_2 = tf.assign(conv2_2,
                            weights['conv2b.weight'].numpy().transpose([3, 2, 1, 0]))
        conv2_2_b = tf.get_variable('conv2_2/conv/bias')
        conv2_2_b = tf.assign(conv2_2_b, weights['conv2b.bias'].numpy())

        conv3_1 = tf.get_variable('conv3_1/conv/kernel')
        conv3_1 = tf.assign(conv3_1,
                            weights['conv3a.weight'].numpy().transpose([3, 2, 1, 0]))
        conv3_1_b = tf.get_variable('conv3_1/conv/bias')
        conv3_1_b = tf.assign(conv3_1_b, weights['conv3a.bias'].numpy())

        conv3_2 = tf.get_variable('conv3_2/conv/kernel')
        conv3_2 = tf.assign(conv3_2,
                            weights['conv3b.weight'].numpy().transpose([3, 2, 1, 0]))
        conv3_2_b = tf.get_variable('conv3_2/conv/bias')
        conv3_2_b = tf.assign(conv3_2_b, weights['conv3b.bias'].numpy())

        conv4_1 = tf.get_variable('conv4_1/conv/kernel')
        conv4_1 = tf.assign(conv4_1,
                            weights['conv4a.weight'].numpy().transpose([3, 2, 1, 0]))
        conv4_1_b = tf.get_variable('conv4_1/conv/bias')
        conv4_1_b = tf.assign(conv4_1_b, weights['conv4a.bias'].numpy())

        conv4_2 = tf.get_variable('conv4_2/conv/kernel')
        conv4_2 = tf.assign(conv4_2,
                            weights['conv4b.weight'].numpy().transpose([3, 2, 1, 0]))
        conv4_2_b = tf.get_variable('conv4_2/conv/bias')
        conv4_2_b = tf.assign(conv4_2_b, weights['conv4b.bias'].numpy())

    # Detector head
    with tf.variable_scope('detector', reuse=tf.AUTO_REUSE):
        convP_1 = tf.get_variable('conv1/conv/kernel')
        convP_1 = tf.assign(convP_1,
                            weights['convPa.weight'].numpy().transpose([3, 2, 1, 0]))
        convP_1_b = tf.get_variable('conv1/conv/bias')
        convP_1_b = tf.assign(convP_1_b, weights['convPa.bias'].numpy())

        convP_2 = tf.get_variable('conv2/conv/kernel')
        convP_2 = tf.assign(convP_2,
                            weights['convPb.weight'].numpy().transpose([3, 2, 1, 0]))
        convP_2_b = tf.get_variable('conv2/conv/bias')
        convP_2_b = tf.assign(convP_2_b, weights['convPb.bias'].numpy())

    # Descriptor head
    with tf.variable_scope('descriptor', reuse=tf.AUTO_REUSE):
        convD_1 = tf.get_variable('conv1/conv/kernel')
        convD_1 = tf.assign(convD_1,
                            weights['convDa.weight'].numpy().transpose([3, 2, 1, 0]))
        convD_1_b = tf.get_variable('conv1/conv/bias')
        convD_1_b = tf.assign(convD_1_b, weights['convDa.bias'].numpy())

        convD_2 = tf.get_variable('conv2/conv/kernel')
        convD_2 = tf.assign(convD_2,
                            weights['convDb.weight'].numpy().transpose([3, 2, 1, 0]))
        convD_2_b = tf.get_variable('conv2/conv/bias')
        convD_2_b = tf.assign(convD_2_b, weights['convDb.bias'].numpy())

# Run and save the tf model
assign_op = [conv1_1, conv1_1_b, conv2_1, conv2_1_b, conv3_1, conv3_1_b,
             conv4_1, conv4_1_b, convP_1, convP_1_b, convD_1, convD_1_b,
             conv1_2, conv1_2_b, conv2_2, conv2_2_b, conv3_2, conv3_2_b,
             conv4_2, conv4_2_b, convP_2, convP_2_b, convD_2, convD_2_b]

sess = tf.Session()
sess.run(tf.initialize_all_variables())
sess.run(assign_op)
saver = tf.train.Saver(save_relative_paths=True)
saver.save(sess, checkpoint_path, write_meta_graph=False)
print("Weights saved to " + checkpoint_path)
