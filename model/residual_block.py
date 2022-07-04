import tensorflow as tf
from tensorflow.keras import layers, initializers
import tensorflow_addons as tfa


class Residual_Block(layers.Layer):
    def __init__(self, out_shape, strides=1, ksize=3, shortcut=True):
        super(Residual_Block, self).__init__()
        self.shortcut = shortcut

        self.conv1_1 = layers.Conv2D(filters=out_shape,
                                     kernel_size=(3, 3), strides=1, padding='same',
                                     name='conv1_1',  use_bias=False)
        self.IN1 = tfa.layers.InstanceNormalization()
        self.ReLU1_1 = layers.LeakyReLU(name='leakyReLU1_1')

        self.conv1_2 = layers.Conv2D(filters=out_shape,
                                     kernel_size=(3, 3), strides=1, padding='same',
                                     name='conv1_2',  use_bias=False)

        self.IN2 = tfa.layers.InstanceNormalization()

    def call(self, inputs):

        x = self.conv1_1(inputs)
        x = self.IN1(x)
        x = self.ReLU1_1(x)

        x = self.conv1_2(x)
        x - self.IN2

        if self.shortcut:
            x = layers.add([x, inputs])

        return x
