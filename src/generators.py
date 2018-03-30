import functools
import tensorflow as tf

class DCGenerator:
    def __init__(self, noise_dim, image_size, num_channel, upsampling_kernel_size=4, h_activation=tf.nn.leaky_relu, o_activation=None):
        """
        noise_dim: original=100
        image_size: original=64. image_size must be either 32 or 64 at this implementation.
        upsampling_kernel_size: defalut=4, original=5.
        h_activation: defalut=tf.nn.leaky_relu, original=relu.
        o_activation: default=identity, original=tanh.
        """
        self.noise_dim = noise_dim
        assert image_size in [32, 64]
        self.image_size = image_size
        self.num_channel = num_channel
        self.upsampling_kernel_size = upsampling_kernel_size
        if h_activation is None: h_activation = tf.identity
        self.h_activation = h_activation
        if o_activation is None: o_activation = tf.identity
        self.o_activation = o_activation

        kernel_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
        bias_initializer = tf.constant_initializer(0.1)
        PartialConv2DTranspose = functools.partial(tf.layers.Conv2DTranspose, padding="same", activation=None, use_bias=True, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)

        self.name = "DCGenerator"

        with tf.variable_scope(self.name):
            self.convtrans1 = PartialConv2DTranspose(filters=1024, kernel_size=4, strides=4, padding="valid") # 4x4
            self.convtrans1.build([None, 1,1, self.noise_dim])
            self.bn1 = tf.layers.BatchNormalization()
            self.bn1.build([None, 4,4, self.convtrans1.filters])

            self.convtrans2 = PartialConv2DTranspose(filters=512, kernel_size=self.upsampling_kernel_size, strides=2) # 8x8
            self.convtrans2.build([None, 4,4, self.convtrans1.filters])
            self.bn2 = tf.layers.BatchNormalization()
            self.bn2.build([None, 8,8, self.convtrans2.filters])

            self.convtrans3 = PartialConv2DTranspose(filters=256, kernel_size=self.upsampling_kernel_size, strides=2) # 16x16
            self.convtrans3.build([None, 8,8, self.convtrans2.filters])
            self.bn3 = tf.layers.BatchNormalization()
            self.bn3.build([None, 16,16, self.convtrans3.filters])

            self.conv_output32 = PartialConv2DTranspose(filters=self.num_channel, kernel_size=self.upsampling_kernel_size, strides=2, bias_initializer=tf.zeros_initializer()) # 32x32
            self.conv_output32.build([None, 16,16, self.convtrans3.filters])

            self.convtrans4 = PartialConv2DTranspose(filters=128, kernel_size=self.upsampling_kernel_size, strides=2) # 32x32
            self.convtrans4.build([None, 16,16, self.convtrans3.filters])
            self.bn4 = tf.layers.BatchNormalization()
            self.bn4.build([None, 32,32, self.convtrans4.filters])

            self.conv_output64 = PartialConv2DTranspose(filters=self.num_channel, kernel_size=self.upsampling_kernel_size, strides=2, bias_initializer=tf.zeros_initializer()) # 64x64
            self.conv_output64.build([None, 32,32, self.convtrans4.filters])

    def __call__(self, noise, is_training):
        with tf.variable_scope(self.name):
            img_noise = tf.reshape(noise, [-1, 1,1, self.noise_dim])

            h1 = self.h_activation(self.bn1(self.convtrans1(img_noise), is_training))
            h2 = self.h_activation(self.bn2(self.convtrans2(h1), is_training))
            h3 = self.h_activation(self.bn3(self.convtrans3(h2), is_training))

            if self.image_size == 32:
                outputs = self.o_activation(self.conv_output32(h3))
            elif self.image_size == 64:
                h4 = self.h_activation(self.bn4(self.convtrans4(h3), is_training))
                outputs = self.o_activation(self.convtrans5(h4))
            else:
                raise Exception("invalid image_size: {}.image_size = {}".format(self, self.image_size))

        return outputs


