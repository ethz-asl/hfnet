import logging
import yaml
import argparse
from pathlib import Path
from pprint import pformat

from hfnet.models import get_model
from hfnet.utils import tools  # noqa: E402
from hfnet.settings import EXPER_PATH, DATA_PATH
import tensorflow as tf


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('export_name', type=str)
    parser.add_argument('--exper_name', type=str)
    args = parser.parse_args()

    export_name = args.export_name
    exper_name = args.exper_name

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    export_dir = Path(EXPER_PATH, 'saved_models', export_name)

    if exper_name:
        assert Path(EXPER_PATH, exper_name).exists()
        with open(Path(EXPER_PATH, exper_name, 'config.yaml'), 'r') as f:
            config['model'] = tools.dict_update(
                yaml.safe_load(f)['model'], config.get('model', {}))
        checkpoint_path = Path(EXPER_PATH, exper_name)
        if config.get('weights', None):
            checkpoint_path = Path(checkpoint_path, config['weights'])
    else:
        checkpoint_path = Path(DATA_PATH, 'weights', config['weights'])
    logging.info(f'Exporting model with configuration:\n{pformat(config)}')

    with get_model(config['model']['name'])(
            data_shape={'image': [None, None, None,
                                  config['model']['image_channels']]},
            **config['model']) as net:

        net.load(str(checkpoint_path))
        print(net.pred_in)
        print(net.pred_out)

        tf.saved_model.simple_save(
                net.sess,
                str(export_dir),
                inputs=net.pred_in,
                outputs=net.pred_out)
        tf.train.write_graph(net.graph, str(export_dir), 'graph.pbtxt')
