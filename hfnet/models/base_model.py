from abc import ABCMeta, abstractmethod
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import itertools


from hfnet.utils.tools import dict_update


class Mode:
    TRAIN = 'train'
    EVAL = 'eval'
    PRED = 'pred'


class BaseModel(metaclass=ABCMeta):
    """Base model class.

    Arguments:
        data: A dictionary of `tf.data.Dataset` objects, can include the keys
            `"training"`, `"validation"`, and `"test"`.
        n_gpus: An integer, the number of GPUs available.
        data_shape: A dictionary, where the keys are the input features of the prediction
            network and the values are the associated shapes. Only required if `data` is
            empty or `None`.
        config: A dictionary containing the configuration parameters.
            Entries `"batch_size"` and `"learning_rate"` are required if `data`is given.

    Models should inherit from this class and implement the following methods:
        `_model`, `_loss`, and `_metrics`.
    Additionally, the following static attributes should be defined:
        input_spec: A dictionary, where the keys are the input features (e.g. `"image"`)
            and the associated values are dictionaries containing `"shape"` (list of
            dimensions, e.g. `[N, H, W, C]` where `None` indicates an unconstrained
            dimension) and `"type"` (e.g. `tf.float32`).
        required_config_keys: A list containing the required configuration entries.
        default_config: A dictionary of potential default configuration values.
    """
    dataset_names = set(['training', 'validation', 'test'])
    required_baseconfig = ['batch_size', 'learning_rate']
    _default_config = {'eval_batch_size': 1}

    @abstractmethod
    def _model(self, inputs, mode, **config):
        """Implements the graph of the model.

        This method is called three times: for training, evaluation and prediction (see
        the `mode` argument) and can return different tensors depending on the mode.
        It is a good practice to support both NCHW (channels first) and NHWC (channels
        last) data formats using a dedicated configuration entry.

        Arguments:
            inputs: A dictionary of input features, where the keys are their names
                (e.g. `"image"`) and the values of type `tf.Tensor`. Same keys as in the
                datasets given during the object instantiation.
            mode: An attribute of the `Mode` class, either `Mode.TRAIN`, `Mode.EVAL` or
                `Mode.PRED`.
            config: A configuration dictionary, given during the object instantiantion.

        Returns:
            A dictionary of outputs, where the keys are their names (e.g. `"logits"`) and
            the values are the corresponding `tf.Tensor`.
        """
        raise NotImplementedError

    @abstractmethod
    def _loss(self, outputs, inputs, **config):
        """Implements the sub-graph computing the training loss.

        This method is called on the outputs of the `_model` method in training mode.

        Arguments:
            outputs: A dictionary, as retuned by `_model` called with `mode=Mode.TRAIN`.
            inputs: A dictionary of input features (see same as for `_model`).
            config: A configuration dictionary.

        Returns:
            A tensor corresponding to the loss to be minimized during training.
        """
        raise NotImplementedError

    @abstractmethod
    def _metrics(self, outputs, inputs, **config):
        """Implements the sub-graph computing the evaluation metrics.

        This method is called on the outputs of the `_model` method in evaluation mode.

        Arguments:
            outputs: A dictionary, as retuned by `_model` called with `mode=Mode.EVAL`.
            inputs: A dictionary of input features (see same as for `_model`).
            config: A configuration dictionary.

        Returns:
            A dictionary of metrics, where the keys are their names (e.g. "`accuracy`")
            and the values are the corresponding `tf.Tensor`.
        """
        raise NotImplementedError

    def __init__(self, data={}, n_gpus=1, data_shape=None, **config):
        self.datasets = data
        self.data_shape = data_shape
        self.n_gpus = n_gpus
        self.graph = tf.get_default_graph()
        self.name = self.__class__.__name__.lower()  # get child name

        # Update config
        self.config = dict_update(
            self.default_config, getattr(self, 'default_config', {}))
        self.config = dict_update(self.config, config)

        required = getattr(self, 'required_config_keys', [])
        if self.datasets:
            required += self.required_baseconfig
        for r in required:
            assert r in self.config, 'Required configuration entry: \'{}\''.format(r)
        assert set(self.datasets) <= self.dataset_names, \
            'Unknown dataset name: {}'.format(set(self.datasets)-self.dataset_names)
        assert n_gpus > 0, 'TODO: CPU-only training is currently not supported.'

        if data_shape is None:
            self.data_shape = {i: s['shape'] for i, s in self.input_spec.items()}

        with tf.variable_scope('', reuse=tf.AUTO_REUSE):
            self._build_graph()

    def _gpu_tower(self, data, mode):
        # Split the batch between the GPUs (data parallelism)
        with tf.device('/cpu:0'):
            with tf.name_scope('{}_data_sharding'.format(mode)):
                batch_size = self.config['batch_size'] if (mode == Mode.TRAIN) \
                        else self.config['eval_batch_size']
                shards = {d: tf.unstack(v, num=batch_size*self.n_gpus, axis=0)
                          for d, v in data.items()}
                shards = [{d: tf.stack(v[i::self.n_gpus]) for d, v in shards.items()}
                          for i in range(self.n_gpus)]

        # Create towers, i.e. copies of the model for each GPU,
        # with their own loss and gradients.
        tower_losses = []
        tower_gradvars = []
        tower_preds = []
        tower_metrics = []
        for i in range(self.n_gpus):
            worker = '/gpu:{}'.format(i)
            device_setter = tf.train.replica_device_setter(
                    worker_device=worker, ps_device='/cpu:0', ps_tasks=1)
            with tf.name_scope('{}_{}'.format(mode, i)) as scope:
                with tf.device(device_setter):
                    net_outputs = self._model(shards[i], mode, **self.config)
                    if mode == Mode.TRAIN:
                        loss = self._loss(net_outputs, shards[i], **self.config)
                        loss += tf.reduce_sum(
                                tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                                                  scope))
                        model_params = tf.trainable_variables()
                        grad = tf.gradients(loss, model_params)
                        tower_losses.append(loss)
                        tower_gradvars.append(zip(grad, model_params))
                        if i == 0:
                            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,
                                                           scope)
                    elif mode == Mode.EVAL:
                        tower_metrics.append(self._metrics(
                            net_outputs, shards[i], **self.config))
                    else:
                        tower_preds.append(net_outputs)

        if mode == Mode.TRAIN:
            return tower_losses, tower_gradvars, update_ops
        elif mode == Mode.EVAL:
            return tower_metrics
        else:
            return tower_preds

    def _train_graph(self, data):
        tower_losses, tower_gradvars, update_ops = self._gpu_tower(data, Mode.TRAIN)

        # Perform the consolidation on CPU
        gradvars = []
        with tf.device('/cpu:0'):
            # Average losses and gradients
            with tf.name_scope('tower_averaging'):
                all_grads = {}
                for grad, var in itertools.chain(*tower_gradvars):
                    if grad is not None:
                        all_grads.setdefault(var, []).append(grad)
                for var, grads in all_grads.items():
                    if len(grads) == 1:
                        avg_grad = grads[0]
                    else:
                        avg_grad = tf.multiply(tf.add_n(grads), 1. / len(grads))
                    gradvars.append((avg_grad, var))
                self.loss = tf.reduce_mean(tower_losses)
                tf.summary.scalar('loss', self.loss)

            # Create optimizer ops
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            lr_val = self.config['learning_rate']
            lr_step = self.config['learning_rate_step']
            if isinstance(lr_step, list):
                assert isinstance(lr_val, list)
                tf.logging.info(f'Using piecewise learning rate {lr_val}'
                             f' at iterations {lr_step}')
                learning_rate = tf.train.piecewise_constant(
                    self.global_step, lr_step, lr_val)
            elif lr_step is not None:
                tf.logging.info('Using exponentially decayed learning rate'
                                f' with value {lr_val:.2E} and step {lr_step}')
                learning_rate = tf.train.exponential_decay(
                    lr_val, self.global_step, lr_step,
                    self.config.get('learning_rate_decay', 0.1), staircase=True)
            else:
                tf.logging.info(
                    f'Using constant learning rate with value {lr_val:.2E}')
                learning_rate = lr_val
            opt = tf.train.RMSPropOptimizer(learning_rate)
            with tf.control_dependencies(update_ops):
                self.trainer = opt.apply_gradients(
                        gradvars, global_step=self.global_step)

    def _eval_graph(self, data):
        tower_metrics = self._gpu_tower(data, Mode.EVAL)
        with tf.device('/cpu:0'):
            self.metrics = {m: tf.reduce_mean(tf.stack([t[m] for t in tower_metrics]))
                            for m in tower_metrics[0]}

    def _pred_graph(self, data):
        with tf.name_scope('pred'):
            with tf.device('/gpu:0'):
                pred_out = self._model(data, Mode.PRED, **self.config)
        self.pred_out = {n: tf.identity(p, name=n) for n, p in pred_out.items()}

    def _build_graph(self):
        # Training and evaluation network, if tf datasets provided
        if self.datasets:
            # Generate iterators for the given tf datasets
            self.dataset_iterators = {}
            with tf.device('/cpu:0'):
                for n, d in self.datasets.items():
                    if n == 'training':
                        train_batch = self.config['batch_size']*self.n_gpus
                        d = d.repeat().batch(train_batch).prefetch(train_batch)
                        self.dataset_iterators[n] = d.make_one_shot_iterator()
                    else:
                        d = d.batch(self.config['eval_batch_size']*self.n_gpus)
                        self.dataset_iterators[n] = d.make_initializable_iterator()
                    output_types = d.output_types
                    output_shapes = d.output_shapes
                    self.datasets[n] = d

                    # Perform compatibility checks with the inputs of the child model
                    for i, spec in self.input_spec.items():
                        assert i in output_shapes
                        tf.TensorShape(output_shapes[i]).assert_is_compatible_with(
                                tf.TensorShape(spec['shape']))

                # Used for input shapes of the prediction network
                if self.data_shape is None:
                    self.data_shape = output_shapes

                # Handle for the feedable iterator
                self.handle = tf.placeholder(tf.string, shape=[])
                iterator = tf.data.Iterator.from_string_handle(
                        self.handle, output_types, output_shapes)
                data = iterator.get_next()

            # Build the actual training and evaluation models
            self._train_graph(data)
            self._eval_graph(data)
            self.summaries = tf.summary.merge_all()

        # Prediction network with feed_dict
        self.pred_in = {i: tf.placeholder(self.input_spec[i]['type'], shape=s, name=i)
                        for i, s in self.data_shape.items()}
        self._pred_graph(self.pred_in)

        # Start session
        sess_config = tf.ConfigProto(device_count={'GPU': self.n_gpus})
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)

        # Register tf dataset handles
        if self.datasets:
            self.dataset_handles = {}
            for n, i in self.dataset_iterators.items():
                self.dataset_handles[n] = self.sess.run(i.string_handle())

        self.sess.run([tf.global_variables_initializer(),
                       tf.local_variables_initializer()])

    def train(self, iterations, validation_interval=100, output_dir=None,
              save_interval=None, checkpoint_path=None, keep_checkpoints=1):
        assert 'training' in self.datasets, 'Training dataset is required.'
        if output_dir is not None:
            train_writer = tf.summary.FileWriter(output_dir)
        if not hasattr(self, 'saver'):
            with tf.device('/cpu:0'):
                self.saver = tf.train.Saver(save_relative_paths=True,
                                            max_to_keep=keep_checkpoints)
        if not self.graph.finalized:
            self.graph.finalize()

        tf.logging.info('Start training')
        for i in range(iterations):
            loss, summaries, _ = self.sess.run(
                    [self.loss, self.summaries, self.trainer],
                    feed_dict={self.handle: self.dataset_handles['training']})

            if save_interval and checkpoint_path and i != 0 and i % save_interval == 0:
                self.save(checkpoint_path)
            if 'validation' in self.datasets and i % validation_interval == 0:
                metrics = self.evaluate('validation', mute=True)
                tf.logging.info(
                        'Iter {:4d}: loss {:.4f}'.format(i, loss) +
                        ''.join([', {} {:.4f}'.format(m, metrics[m]) for m in metrics]))

                if output_dir is not None:
                    train_writer.add_summary(summaries, i)
                    metrics_summaries = tf.Summary(value=[
                        tf.Summary.Value(tag=m, simple_value=v)
                        for m, v in metrics.items()])
                    train_writer.add_summary(metrics_summaries, i)
        tf.logging.info('Training finished')

    def predict(self, data, keys='*', batch=False):
        assert set(data.keys()) >= set(self.data_shape.keys())
        if isinstance(keys, str):
            if keys == '*':
                op = self.pred_out  # just gather all outputs
            else:
                op = self.pred_out[keys]
        else:
            op = {k: self.pred_out[k] for k in keys}
        if not batch:  # add batch dimension
            data = {d: [v] for d, v in data.items()}
        feed = {self.pred_in[i]: data[i] for i in self.data_shape}
        pred = self.sess.run(op, feed_dict=feed)
        if not batch:  # remove batch dimension
            if isinstance(pred, dict):
                pred = {p: v[0] for p, v in pred.items()}
            else:
                pred = pred[0]
        return pred

    def evaluate(self, dataset, max_iterations=None, mute=False):
        assert dataset in self.datasets
        self.sess.run(self.dataset_iterators[dataset].initializer)

        if not mute:
            tf.logging.info('Starting evaluation of dataset \'{}\''.format(dataset))
            if max_iterations:
                pbar = tqdm(total=max_iterations, ascii=True)
        i = 0
        metrics = []
        while True:
            try:
                metrics.append(self.sess.run(self.metrics,
                               feed_dict={self.handle: self.dataset_handles[dataset]}))
            except tf.errors.OutOfRangeError:
                break
            if max_iterations:
                i += 1
                if not mute:
                    pbar.update(1)
                if i == max_iterations:
                    break
        if not mute:
            tf.logging.info('Finished evaluation')
            if max_iterations:
                pbar.close()

        # List of dicts to dict of lists
        metrics = dict(zip(metrics[0], zip(*[m.values() for m in metrics])))
        metrics = {m: np.nanmean(metrics[m], axis=0) for m in metrics}
        return metrics

    def _checkpoint_var_search(self, checkpoint_path):
        reader = tf.train.NewCheckpointReader(checkpoint_path)
        saved_shapes = reader.get_variable_to_shape_map()
        model_names = tf.model_variables()  # Used by tf.slim layers
        if not len(tf.model_variables()):
            model_names = tf.global_variables()  # Fallback when slim is not used
        model_names = set([v.name.split(':')[0] for v in model_names])
        checkpoint_names = set(saved_shapes.keys())
        found_names = model_names & checkpoint_names
        missing_names = model_names - checkpoint_names
        shape_conflicts = set()
        restored = []
        with tf.variable_scope('', reuse=True):
            for name in found_names:
                var = tf.get_variable(name)
                var_shape = var.get_shape().as_list()
                if var_shape == saved_shapes[name]:
                    restored.append(var)
                else:
                    shape_conflicts.add(name)
        found_names -= shape_conflicts
        return (restored, sorted(found_names),
                sorted(missing_names), sorted(shape_conflicts))

    def load(self, checkpoint_path, flexible_restore=True):
        if tf.gfile.IsDirectory(checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
            if checkpoint_path is None:
                raise ValueError('Checkpoint directory is empty.')
        if flexible_restore:
            var_list, found, missing, conflicts = self._checkpoint_var_search(
                    checkpoint_path)
            tf.logging.info('Restoring variables: \n\t{}'.format(
                '\n\t'.join(found)))
            if len(missing) > 0:
                tf.logging.info('Variables not found in checkpoint: \n\t{}'.format(
                    '\n\t'.join(missing)))
            if len(conflicts) > 0:
                tf.logging.info('Variables with incompatible shapes: \n\t{}'.format(
                    '\n\t'.join(conflicts)))
        else:
            var_list = None
        with tf.device('/cpu:0'):
            saver = tf.train.Saver(var_list=var_list, save_relative_paths=True)
        saver.restore(self.sess, checkpoint_path)

    def save(self, checkpoint_path):
        step = self.sess.run(self.global_step)
        tf.logging.info('Saving checkpoint for iteration #{}'.format(step))
        self.saver.save(self.sess, checkpoint_path, write_meta_graph=False,
                        global_step=step)

    def close(self):
        self.sess.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
