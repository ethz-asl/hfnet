import argparse
import os
import time
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants

tf.contrib.resampler  # import C++ op


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('--input_size', type=int, default=960)
    parser.add_argument('--global_head', action='store_true')
    parser.add_argument('--local_head', action='store_true')
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--use_cpu', action='store_true')
    args = parser.parse_args()

    b, h, w = 1, args.input_size, int(args.input_size*3/4)

    output_names = []
    if args.global_head:
        output_names += ['global_descriptor']
    if args.local_head:
        output_names += ['keypoints', 'scores', 'local_descriptors']
    assert len(output_names) > 0

    measure_iter = args.iterations
    warmup_iter = 100

    if args.use_cpu:
        device = '/cpu:0'
    else:
        gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        assert len(gpus) > 0
        device = '/device:GPU:{}'.format(gpus[0])

    with tf.Session(graph=tf.Graph()) as sess:
        with tf.device(device):
            image = tf.random_uniform([b, h, w, 1], dtype=tf.float32)
        tf.saved_model.loader.load(
            sess, [tag_constants.SERVING], args.model,
            clear_devices=False, input_map={'image:0': image})

        graph = tf.get_default_graph()
        outputs = [graph.get_tensor_by_name(n+':0') for n in output_names]

        for _ in range(warmup_iter):
            out = sess.run(outputs)
            del out

        start_time = time.time()
        for _ in range(measure_iter):
            out = sess.run(outputs)
            del out

        duration = time.time() - start_time

        print(f'Total: {duration:.4f}, per batch: {duration/measure_iter:.4f}')
