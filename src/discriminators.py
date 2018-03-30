import functools
import tensorflow as tf

class DCDiscriminator:
    def __init__(self, image_size, num_channel, downsampling_kernel_size=4, h_activation=tf.nn.leaky_relu, o_activation=None, global_average_pooling=False, batch_normalization=False):
        """
        image_size: original=64. image_size must be either 32 or 64 at this implementation.
        downsampling_kernel_size: defalut=4, original=5.
        h_activation: defalut=original=tf.nn.leaky_relu
        o_activation: default=None(identity)
        global_average_pooling: default=False, original=True.
        batch_normalization: default=False, original=True. using BN in the discriminator of WGAN-GP may cause unconvergence.
        """
        assert image_size in [32, 64]
        assert isinstance(batch_normalization, bool)
        self.image_size = image_size
        self.num_channel = num_channel
        self.downsampling_kernel_size = downsampling_kernel_size
        self.h_activation = h_activation
        self.o_activation = o_activation
        self._global_average_pooling = global_average_pooling # read-only property
        self._batch_normalization = batch_normalization # read-only property

        def cond_build_batch_norm(input_shape):
            if self.batch_normalization:
                bn = tf.layers.BatchNormalization()
                bn.build(input_shape)
                return bn
            else:
                identity = lambda inputs, is_training=False: inputs
                return identity

        kernel_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
        bias_initializer = tf.constant_initializer(0.1)
        PartialConv2D = functools.partial(tf.layers.Conv2D, padding="same", activation=None, use_bias=True, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)
        hidden_size_to_filters = {32:64, 16:128, 8:256, 4:512}

        self.name = "DCDiscriminator"
        with tf.variable_scope(self.name):
            input_size, output_size = 64, 32
            self.conv_input64to32 = PartialConv2D(filters=hidden_size_to_filters[output_size], kernel_size=self.downsampling_kernel_size, strides=2) # 64x64=>32x32
            self.conv_input64to32.build([None, input_size,input_size, self.num_channel])
            self.bn_input64to32 = cond_build_batch_norm([None, output_size,output_size, hidden_size_to_filters[output_size]])

            input_size, output_size = 32, 16
            self.conv_hidden32to16 = PartialConv2D(filters=hidden_size_to_filters[output_size], kernel_size=self.downsampling_kernel_size, strides=2) # 32x32=>16x16
            self.conv_hidden32to16.build([None, input_size,input_size, hidden_size_to_filters[input_size]])
            self.bn_hidden32to16 = cond_build_batch_norm([None, output_size,output_size, hidden_size_to_filters[output_size]])

            input_size, output_size = 32, 16
            self.conv_input32to16 = PartialConv2D(filters=hidden_size_to_filters[output_size], kernel_size=self.downsampling_kernel_size, strides=2) # 32x32=>16x16
            self.conv_input32to16.build([None, input_size,input_size, self.num_channel])
            self.bn_input32to16 = cond_build_batch_norm([None, output_size,output_size, hidden_size_to_filters[output_size]])

            input_size, output_size = 16, 8
            self.conv_hidden16to8 = PartialConv2D(filters=hidden_size_to_filters[output_size], kernel_size=self.downsampling_kernel_size, strides=2) # 16x16=>8x8
            self.conv_hidden16to8.build([None, input_size,input_size, hidden_size_to_filters[input_size]])
            self.bn_hidden16to8 = cond_build_batch_norm([None, output_size,output_size, hidden_size_to_filters[output_size]])

            input_size, output_size = 8, 4
            self.conv_hidden8to4 = PartialConv2D(filters=hidden_size_to_filters[output_size], kernel_size=self.downsampling_kernel_size, strides=2) # 16x16=>8x8
            self.conv_hidden8to4.build([None, input_size,input_size, hidden_size_to_filters[input_size]])
            self.bn_hidden8to4 = cond_build_batch_norm([None, output_size,output_size, hidden_size_to_filters[output_size]])

            input_size = 4
            input_filters = hidden_size_to_filters[input_size]
            if not self.global_average_pooling:
                input_filters *= input_size * input_size
            self.fc_output = tf.layers.Dense(units=1, activation=None, use_bias=False, kernel_initializer=kernel_initializer)
            self.fc_output.build([None, input_filters])

    @property
    def global_average_pooling(self): # -> bool (getter)
        return self._global_average_pooling
    @property
    def batch_normalization(self): # -> bool (getter)
        return self._batch_normalization

    def __call__(self, image, is_training):
        assert self.image_size in [32, 64]

        with tf.variable_scope(self.name):
            # 64x64 => 32x32
            if self.image_size > 64:
                raise Exception("invalid image_size: {}.image_size = {}".format(self, self.image_size))
            elif self.image_size == 64:
                h_32 = self.h_activation(self.bn_input64to32(self.conv_input64to32(image), is_training))

            # 32x32 => 16x16
            if self.image_size > 32:
                h_16 = self.h_activation(self.bn_hidden32to16(self.conv_hidden32to16(h_32), is_training))
            elif self.image_size == 32:
                h_16 = self.h_activation(self.bn_input32to16(self.conv_input32to16(image), is_training))

            # 16x16 => 8x8
            h_8 = self.h_activation(self.bn_hidden16to8(self.conv_hidden16to8(h_16), is_training))

            # 8x8 => 4x4
            h_4 = self.h_activation(self.bn_hidden8to4(self.conv_hidden8to4(h_8), is_training))

            if self.global_average_pooling:
                h_flat = tf.reduce_mean(h_4, axis=(1,2))
            else:
                h_flat = tf.layers.Flatten()(h_4)
            outputs = self.fc_output(h_flat)

        return outputs


