import tensorflow as tf
from tensorflow.keras import layers, initializers
import tensorflow_addons as tfa
from model.residual_block import Residual_Block
from tensorflow.keras.utils import plot_model


class Identification_Discriminator(tf.keras.Model):

    # def __init__(self, num_class: int, num_pei_channel: int, view_dim: int):
    def __init__(self, ch):
        super(Identification_Discriminator, self).__init__()

        self.conv0 = layers.Conv2D(filters=ch*2, kernel_size=(7, 7),
                                   strides=1, padding='same', name='conv0', use_bias=False)
        self.IN0 = tfa.layers.InstanceNormalization()
        self.ReLU0 = layers.ReLU()

        self.conv1 = layers.Conv2D(filters=ch*2, kernel_size=(7, 7),
                                   strides=1, padding='same', name='conv0', use_bias=False)
        self.IN1 = tfa.layers.InstanceNormalization()
        self.ReLU1 = layers.ReLU()

        self.res0 = Residual_Block(ch*4, ksize=3, shortcut=True)
        self.res1 = Residual_Block(ch*4, ksize=3, shortcut=True)
        self.res2 = Residual_Block(ch*4, ksize=3, shortcut=True)
        self.res3 = Residual_Block(ch*4, ksize=3, shortcut=True)
        self.res4 = Residual_Block(ch*4, ksize=3, shortcut=True)
        self.res5 = Residual_Block(ch*4, ksize=3, shortcut=True)
        

        # self.flatten = layers.Flatten()
        # self.f1 = layers.Dense(units=256, name='F1')
        # self.L2_normalize = layers.Lambda(
        #     lambda x: tf.math.l2_normalize(x, axis=1))
        # self.tanh = tf.keras.activations.tanh

    def call(self, input):
        x = self.conv0(input)
        x = self.IN0(x)
        x = self.ReLU0(x)
        x = self.conv1(x)
        x = self.IN1(x)
        x = self.ReLU1(x)

        x = self.res0(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        # x = self.flatten(x)
        # x = self.f1(x)
        # x = self.L2_normalize(x)
        return x

    def model(self, inputsize: int) -> tf.keras.models:
        input = tf.keras.Input(
            shape=(inputsize[0], inputsize[1], inputsize[2]), name='input_layer')
        model = tf.keras.models.Model(inputs=input, outputs=self.call(input))
        plot_model(model, to_file='model.png', show_shapes=True)
        return model
