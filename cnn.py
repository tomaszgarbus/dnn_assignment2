import tensorflow as tf
from typing import Optional, List, Tuple

from loader import Loader
from constants import INPUT_SIZE, DOWNCONV_FILTERS, UPCONV_FILTERS

FilterDesc = Tuple[int, List[int]]


# TODO: proper weights initialization
class UNet:
    loader: Loader = Loader()
    mb_size: int = 10
    learning_rate: float = 0.3
    input_size: [int, int] = INPUT_SIZE
    downconv_filters: List[FilterDesc] = DOWNCONV_FILTERS
    upconv_filters: List[FilterDesc] = UPCONV_FILTERS

    downconv_layers = []
    downpool_layers = []
    upconv_layers = []

    def __init__(self,
                 mb_size: Optional[int] = None,
                 learning_rate: Optional[float] = None,
                 input_size: Optional[List[int]] = None,
                 downconv_filters: Optional[List[FilterDesc]] = None,
                 upconv_filters: Optional[List[FilterDesc]] = None):
        if mb_size:
            self.mb_size = mb_size
        if learning_rate:
            self.learning_rate = learning_rate
        if input_size:
            self.input_size = input_size
        if downconv_filters:
            self.downconv_filters = downconv_filters
        if upconv_filters:
            self.upconv_filters = upconv_filters

        self._create_model()
        pass

    def _add_downconv_layers(self):
        signal = self.x

        for layer_no in range(len(self.downconv_filters)):
            filters_count, kernel_size = self.downconv_filters[layer_no]
            # Convolutional layer
            downconv_layer = tf.layers.conv2d(signal,
                                              filters=filters_count,
                                              kernel_size=kernel_size,
                                              padding='same',
                                              activation=tf.nn.relu,
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
            # Convolutional layer
            upconv_layer = tf.layers.conv2d(signal,
                                            filters=filters_count,
                                            kernel_size=kernel_size,
                                            padding='same',
                                            activation=tf.nn.relu,
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
                print(cur_shape, new_shape)
                uppooling_layer = tf.image.resize_nearest_neighbor(images=upconv_layer,
                                                                   size=new_shape[1:3])
                signal = uppooling_layer
            else:
                signal = upconv_layer

        self.output = signal
        print(signal.get_shape())

    def _add_training_objectives(self):
        self.loss = tf.reduce_mean(self.y - self.output)
        self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

    def _create_model(self):
        self.x = tf.placeholder(tf.float32, [self.mb_size] + self.input_size + [3])
        self.y = tf.placeholder(tf.float32, [self.mb_size] + self.input_size + [1])

        self._add_downconv_layers()
        self._add_upconv_layers()

        self._add_training_objectives()


if __name__ == '__main__':
    net = UNet()
    pass
