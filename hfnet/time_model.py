import argparse
import time
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    args = parser.parse_args()

    b, h, w = 1, 480, 640
    output_names = ['global_descriptor']
    # output_names = ['keypoints', 'scores', 'local_descriptor_map']
    warmup_iter = 100
    measure_iter = 1000

    with tf.Session(graph=tf.Graph()) as sess:
        with tf.device('/device:GPU:0'):
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
