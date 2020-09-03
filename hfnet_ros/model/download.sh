#!/bin/sh
echo "Download hfnet model..."
wget http://robotics.ethz.ch/~asl-datasets/2019_CVPR_hierarchical_localization/hfnet_tf.tar.gz
echo "Download suscess! start extract..."
tar -xavf hfnet_tf.tar.gz

rm -rf hfnet
rm *.gz

echo "Done!"
