import tensorflow as tf
import sys
from tensorflow.python.platform import gfile

from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.util import compat
import tensorflow.contrib
from hfnet.settings import EXPER_PATH

# Run this script in order to create a log file for the tensorboard

with tf.Session() as sess:
    model_filename = EXPER_PATH + '/hfnet_mobv3_retrain_5000/saved_model.pb'
    with gfile.FastGFile(model_filename, 'rb') as f:
        data = compat.as_bytes(f.read())
        sm = saved_model_pb2.SavedModel()
        sm.ParseFromString(data)
        #print(sm)
        if 1 != len(sm.meta_graphs):
      	    print('More than one graph found. Not sure which to write')
      	    sys.exit(1)

      	#graph_def = tf.GraphDef()
        #graph_def.ParseFromString(sm.meta_graphs[0])
        g_in = tf.import_graph_def(sm.meta_graphs[0].graph_def)
LOGDIR= EXPER_PATH + '/hfnet_mobv3_retrain_5000/log'
print(sess.graph)
train_writer = tf.summary.FileWriter(LOGDIR)
train_writer.add_graph(sess.graph)
