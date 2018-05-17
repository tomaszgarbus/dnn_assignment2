import tensorflow as tf
from typing import Optional, List, Tuple
from math import sqrt
import numpy as np
import time
import logging

from loader import Loader
from constants import INPUT_SIZE, DOWNCONV_FILTERS, UPCONV_FILTERS, NUM_LABELS, VAL_SIZE

FilterDesc = Tuple[int, List[int]]


# TODO: data augmentation (horizontal flips: done)
# TODO: checkpoints
# TODO: resizing back to original resolution after making predictions
class UNet:
    loader: Loader = Loader()
    mb_size = 1
    learning_rate = 0.3
    lr_decay = 5000
    nb_epochs = 100000
    input_size = INPUT_SIZE
    downconv_filters = DOWNCONV_FILTERS
    upconv_filters = UPCONV_FILTERS

    downconv_layers = []
    downpool_layers = []
    upconv_layers = []

    def __init__(self,
                 sess,
                 mb_size: Optional[int] = None,
                 learning_rate: Optional[float] = None,
                 lr_decay: Optional[int] = None,
                 nb_epochs: Optional[int] = None,
                 input_size: Optional[List[int]] = None,
                 downconv_filters: Optional[List[FilterDesc]] = None,
                 upconv_filters: Optional[List[FilterDesc]] = None):
        self.sess = sess
        if mb_size:
            self.mb_size = mb_size
        if learning_rate:
            self.learning_rate = learning_rate
        if lr_decay:
            self.lr_decay = lr_decay
        if nb_epochs:
            self.nb_epochs = nb_epochs
        if input_size:
            self.input_size = input_size
        if downconv_filters:
            self.downconv_filters = downconv_filters
        if upconv_filters:
            self.upconv_filters = upconv_filters

        # Initialize logging.
        self.logger = logging.Logger("main_logger", level=logging.INFO)
        log_file = 'log' + str(time.ctime()) + '.txt'
        formatter = logging.Formatter(
            fmt='{message}',
            style='{'
        )
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

        self._create_model()

    def _add_downconv_layers(self):
        signal = self.x

        for layer_no in range(len(self.downconv_filters)):
            filters_count, kernel_size = self.downconv_filters[layer_no]
            # Weights initialization (std. dev = sqrt(2 / N))
            cur_shape = tuple(map(int, signal.get_shape()))
            inputs = cur_shape[1] * cur_shape[2] * cur_shape[3]
            w_init = tf.initializers.random_normal(stddev=sqrt(2 / inputs))
            # Convolutional layer
            downconv_layer = tf.layers.conv2d(signal,
                                              filters=filters_count,
                                              kernel_size=kernel_size,
                                              padding='same',
                                              activation=tf.nn.relu,
                                              kernel_initializer=w_init,
                                              use_bias=False)  # Bias not needed with batch normalization
            self.downconv_layers.append(downconv_layer)
            # Batch normalization layer
            batch_norm = tf.layers.batch_normalization(downconv_layer)
            # Downpooling layer
            downpooling_layer = tf.layers.max_pooling2d(batch_norm,
                                                        pool_size=[2, 2],
                                                        strides=[2, 2],
                                                        padding='same')
            self.downpool_layers.append(downpooling_layer)
            signal = downpooling_layer

    def _add_upconv_layers(self):
        signal = self.downpool_layers[-1]
        for layer_no in range(len(self.upconv_filters)):
            filters_count, kernel_size = self.upconv_filters[layer_no]
            # Weights initialization (std. dev = sqrt(2 / N))
            cur_shape = tuple(map(int, signal.get_shape()))
            inputs = cur_shape[1] * cur_shape[2] * cur_shape[3]
            w_init = tf.initializers.random_normal(stddev=sqrt(2 / inputs))
            # Convolutional layer
            upconv_layer = tf.layers.conv2d(signal,
                                            filters=filters_count,
                                            kernel_size=kernel_size,
                                            padding='same',
                                            activation=tf.nn.relu,
                                            kernel_initializer=w_init,
                                            use_bias=False)
            # Batch normalization layer
            batch_norm = tf.layers.batch_normalization(upconv_layer)
            # Concatenate with respective downconv
            if layer_no and layer_no < len(self.downconv_layers):
                upconv_concat = tf.concat([batch_norm, self.downconv_layers[-layer_no]], axis=3)
                upconv_layer = upconv_concat
            else:
                upconv_layer = batch_norm
            # Upsampling layer
            if layer_no < len(self.downconv_layers):
                cur_shape = tuple(map(int, upconv_layer.get_shape()))
                new_shape = (cur_shape[0], cur_shape[1] * 2, cur_shape[2] * 2, cur_shape[3])
                self.logger.info((cur_shape, new_shape))
                uppooling_layer = tf.image.resize_nearest_neighbor(images=upconv_layer,
                                                                   size=new_shape[1:3])
                signal = uppooling_layer
            else:
                signal = upconv_layer

        self.output = signal
        self.logger.info(signal.get_shape())

    def _add_training_objectives(self):
        self.loss = tf.reduce_mean(tf.losses.mean_squared_error(self.y, self.output))
        self.preds = tf.argmax(self.output, axis=3)
        self.labels = tf.argmax(self.y, axis=3)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.preds, self.labels), tf.float32))
        self.train_op = tf.train.MomentumOptimizer(self.learning_rate, momentum=0.9).minimize(self.loss)

    def _create_model(self):
        self.x = tf.placeholder(tf.float32, [self.mb_size] + self.input_size + [3])
        self.y = tf.placeholder(tf.float32, [self.mb_size] + self.input_size + [NUM_LABELS])

        self._add_downconv_layers()
        self._add_upconv_layers()

        self._add_training_objectives()

        # Initialize variables.
        tf.global_variables_initializer().run()

    def _train_on_batch(self):
        batch_x, batch_y = self.loader.prepare_batch(self.mb_size)
        results = self.sess.run([self.loss,
                                 self.accuracy,
                                 self.train_op,
                                 self.preds,
                                 self.labels],
                                feed_dict={self.x: batch_x, self.y: batch_y})
        return results

    def _test_on_batch(self, img_no_first, img_no_last):
        batch_x, batch_y = self.loader.validation_batch(img_no_first, img_no_last)
        results = self.sess.run([self.loss, self.accuracy],
                                feed_dict={self.x: batch_x, self.y: batch_y})
        return results

    def train(self):
        accs = []
        for epoch_no in range(self.nb_epochs):
            loss, acc, _, preds, labels = self._train_on_batch()
            accs.append(acc)
            if epoch_no % 100 == 0:
                self.logger.info('{0}: epoch {1}/{2}: loss: {3}, acc: {4}, mean_acc: {5}'
                      .format(time.ctime(), epoch_no, self.nb_epochs, loss, acc, np.mean(accs[-1000:])))
            if epoch_no % 1000 == 0 or epoch_no == self.nb_epochs - 1:
                net.loader.show_image_or_labels(preds[0])
                net.loader.show_image_or_labels(labels[0])
            if epoch_no and epoch_no % 18000 == 0:
                self.validate()
            if epoch_no and epoch_no % self.lr_decay == 0:
                self.learning_rate /= 2.

    def validate(self):
        accs = []
        for i in range(VAL_SIZE):
            for flip in [False, True]:
                loss, acc = self._test_on_batch(i, i, flip=flip)
                accs.append(acc)
        self.logger.info("Validation accuracy: {0}".format(np.mean(accs)))


if __name__ == '__main__':
    with tf.Session() as sess:
        net = UNet(sess)
        net.train()
    pass
