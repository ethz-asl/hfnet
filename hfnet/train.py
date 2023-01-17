import logging
import yaml
import os
import argparse
import numpy as np
from contextlib import contextmanager
from json import dumps as pprint

from hfnet.datasets import get_dataset
from hfnet.models import get_model
from hfnet.utils.stdout_capturing import capture_outputs
from hfnet.settings import EXPER_PATH, DATA_PATH

logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
import tensorflow as tf  # noqa: E402


def train(config, n_iter, output_dir, checkpoint_name='model.ckpt'):
    checkpoint_path = os.path.join(output_dir, checkpoint_name)
    with _init_graph(config) as net:
        if 'weights' in config:
            net.load(os.path.join(DATA_PATH, 'weights', config['weights']))
        elif 'weights_exper' in config:
            net.load(os.path.join(EXPER_PATH, config['weights_exper']))
        try:
            net.train(n_iter, output_dir=output_dir,
                      validation_interval=config.get('validation_interval', 100),
                      save_interval=config.get('save_interval', None),
                      checkpoint_path=checkpoint_path,
                      keep_checkpoints=config.get('keep_checkpoints', 1))
        except KeyboardInterrupt:
            logging.info('Got Keyboard Interrupt, saving model and closing.')
        net.save(checkpoint_path)


def set_seed(seed):
    tf.set_random_seed(seed)
    np.random.seed(seed)


@contextmanager
def _init_graph(config, with_dataset=False):
    set_seed(config.get('seed', int.from_bytes(os.urandom(4), byteorder='big')))
    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    logging.info('Number of GPUs detected: {}'.format(n_gpus))

    dataset = get_dataset(config['data']['name'])(**config['data'])
    #print(**config['data'])
    #print(dataset)
    model = get_model(config['model']['name'])(
            data=dataset.get_tf_datasets(), n_gpus=n_gpus, **config['model'])
    model.__enter__()
    if with_dataset:
        yield model, dataset
    else:
        yield model
    model.__exit__()
    tf.reset_default_graph()


def _cli_train(config, output_dir):
    assert 'train_iter' in config

    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    train(config, config['train_iter'], output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('exper_name', type=str)

    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    output_dir = os.path.join(EXPER_PATH, args.exper_name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    with capture_outputs(os.path.join(output_dir, 'log')):
        _cli_train(config, output_dir)
