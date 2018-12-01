import yaml
import argparse
from pathlib import Path
import tensorflow as tf

from hfnet.models import get_model
from hfnet.settings import EXPER_PATH, DATA_PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('export_name', type=str)
    parser.add_argument('--exper_name', type=str)
    args = parser.parse_args()

    export_name = args.export_name
    exper_name = args.exper_name

    with open(args.config, 'r') as f:
        config = yaml.load(f)

    export_dir = Path(EXPER_PATH, 'saved_models', export_name)

    if exper_name:
        assert Path(EXPER_PATH, exper_name).exists()
        checkpoint_path = Path(EXPER_PATH, exper_name)
        if config.get('weights', None):
            checkpoint_path = Path(checkpoint_path, config['weights'])
    else:
        checkpoint_path = Path(DATA_PATH, 'weights', config['weights'])

    with get_model(config['model']['name'])(
            data_shape={'image': [None, None, None,
                                  config['model']['image_channels']]},
            **config['model']) as net:

        net.load(str(checkpoint_path))

        tf.saved_model.simple_save(
                net.sess,
                str(export_dir),
                inputs=net.pred_in,
                outputs=net.pred_out)
        tf.train.write_graph(net.graph, str(export_dir), 'graph.pbtxt')
