import tensorflow as tf
from tensorflow.keras import layers, initializers
import tensorflow_addons as tfa
from model.residual_block import Residual_Block
from tensorflow.keras.utils import plot_model


class Generator(tf.keras.Model):

    # def __init__(self, num_class: int, num_pei_channel: int, view_dim: int):
    def __init__(self, ch):
        super(Generator, self).__init__()
        self.concatenate = layers.concatenate
        self.conv0 = layers.Conv2D(filters=ch*2, kernel_size=(7, 7),
                                   strides=1, padding='same', name='conv0', use_bias=False)
        self.IN0 = tfa.layers.InstanceNormalization()
        self.ReLU0 = layers.ReLU()

        self.conv1 = layers.Conv2D(filters=ch*4, kernel_size=(4, 4),
                                   strides=2, padding='same', name='conv1', use_bias=False)
        self.IN1 = tfa.layers.InstanceNormalization()
        self.ReLU1 = layers.ReLU()

        self.conv2 = layers.Conv2D(filters=ch*8, kernel_size=(4, 4),
                                   strides=2, padding='same', name='conv2', use_bias=False)
        self.IN2 = tfa.layers.InstanceNormalization()
        self.ReLU2 = layers.ReLU()

        self.res0 = Residual_Block(ch*8, ksize=3, shortcut=True)
        self.res1 = Residual_Block(ch*8, ksize=3, shortcut=True)
        self.res2 = Residual_Block(ch*8, ksize=3, shortcut=True)
        self.res3 = Residual_Block(ch*8, ksize=3, shortcut=True)
        self.res4 = Residual_Block(ch*8, ksize=3, shortcut=True)
        self.res5 = Residual_Block(ch*8, ksize=3, shortcut=True)

        self.Deconv1 = layers.Conv2DTranspose(filters=ch*4, kernel_size=(4, 4),
                                              strides=2, padding='same', name='deconv1', use_bias=False)
        self.Deconv_IN1 = tfa.layers.InstanceNormalization()
        self.Deconv_ReLU1 = layers.ReLU()

        self.Deconv2 = layers.Conv2DTranspose(filters=ch*2, kernel_size=(4, 4),
                                              strides=2, padding='same', name='deconv2', use_bias=False)
        self.Deconv_IN2 = tfa.layers.InstanceNormalization()
        self.Deconv_ReLU2 = layers.ReLU()

        self.output_conv = layers.Conv2D(filters=3, kernel_size=(7, 7),
                                         strides=1, padding='same', name='output_conv', use_bias=False)
        self.output_conv_IN = tfa.layers.InstanceNormalization()
        self.tanh = layers.Activation('tanh')

    def call(self, input):
        x = self.concatenate(input, axis=-1)

        x = self.conv0(x)
        x = self.IN0(x)
        x = self.ReLU0(x)

        x = self.conv1(x)
        x = self.IN1(x)
        x = self.ReLU1(x)

        x = self.conv2(x)
        x = self.IN2(x)
        x = self.ReLU2(x)

        x = self.res0(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)

        x = self.Deconv1(x)
        x = self.Deconv_IN1(x)
        x = self.Deconv_ReLU1(x)

        x = self.Deconv2(x)
        x = self.Deconv_IN2(x)
        x = self.Deconv_ReLU2(x)

        x = self.output_conv(x)
        x = self.output_conv_IN(x)
        x = self.tanh(x)

        return x

    def model(self, inputsize: int, labelsize: int) -> tf.keras.models:

        input = tf.keras.Input(
            shape=(inputsize[0], inputsize[1], inputsize[2]), name='input_layer')
        label = tf.keras.Input(
            shape=(inputsize[0], inputsize[1], labelsize), name='label_layer')
        model = tf.keras.models.Model(
            inputs=[input, label], outputs=self.call([input, label]))

        plot_model(model, to_file='Generator.png', show_shapes=True)
        return model
