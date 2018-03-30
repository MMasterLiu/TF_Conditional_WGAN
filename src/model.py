import os
import functools
import tensorflow as tf
import numpy as np

from . import generators
from . import discriminators

class Model:
    def __init__(self, num_channel, num_class, noise_dim=100):
        self.noise_dim = noise_dim
        self.num_channel = num_channel
        self.num_class = num_class
        self.image_size = 32
        with tf.variable_scope("Generator"):
            self.generator = generators.DCGenerator(noise_dim=self.noise_dim+self.num_class, image_size=self.image_size, num_channel=self.num_channel, o_activation=tf.nn.tanh)
        with tf.variable_scope("Discriminator"):
            self.discriminator = discriminators.DCDiscriminator(image_size=self.image_size, num_channel=self.num_channel+self.num_class)

        self.is_training = tf.constant(True)

    def build_network_outputs(self, noise, real_image, fake_labels=None, real_labels=None):
        assert (fake_labels is None) == (real_labels is None)
        use_condition = (fake_labels is not None)
        assert (self.num_class != 0) == use_condition

        with tf.variable_scope("Generator"):
            self.gen_image = self.generator(noise, self.is_training)
        self.real_image = real_image

        if use_condition:
            self.disc_fake_inputs = self._condition(self.gen_image, fake_labels)
            self.disc_real_inputs = self._condition(self.real_image, real_labels)
        else:
            self.disc_fake_inputs = self.gen_image
            self.disc_real_inputs = self.real_image

        with tf.variable_scope("Discriminator"):
            self.gen_disc = self.discriminator(self.disc_fake_inputs, self.is_training)
            self.real_disc = self.discriminator(self.disc_real_inputs, self.is_training)

    def build_losses(self, use_gradient_penalty=True, lambda_gradient_penaly=10.0):
        mean_real_disc = tf.reduce_mean(self.real_disc)
        mean_gen_disc = tf.reduce_mean(self.gen_disc)
        self.w_distance = mean_real_disc - mean_gen_disc

        self.disc_loss = -self.w_distance
        if use_gradient_penalty:
            batch_size = tf.shape(self.real_image)[0]
            internal_ratio = tf.random_uniform([batch_size, 1,1,1], 0.0, 1.0)
            internally = internal_ratio * self.disc_fake_inputs + (1.0-internal_ratio) * self.disc_real_inputs
            with tf.variable_scope("Discriminator"):
                internal_disc = self.discriminator(internally, self.is_training)
            gradient = tf.gradients(internal_disc, [internally])[0]
            gradient_norm = tf.norm(tf.layers.Flatten()(gradient), axis=1)
            self.disc_loss += lambda_gradient_penaly * tf.reduce_mean(tf.pow(gradient_norm-1.0, 2.0))
        self.gen_loss = -mean_gen_disc

    def _condition(self, image, labels):
        one_hot = tf.one_hot(labels, depth=self.num_class)
        image_one_hot = tf.tile(one_hot[:, tf.newaxis,tf.newaxis, :], [1, self.image_size,self.image_size, 1])
        conditioned = tf.concat([image, image_one_hot], axis=3)
        return conditioned

    def _random_noise(self, batch_size, default_labels=None):
        #r_free_noise = tf.random_normal([batch_size, self.noise_dim])
        #r = tf.norm(r_free_noise, axis=1, keep_dims=True)
        #sphere_noise = r_free_noise / tf.maximum(r, 1e-20)

        #uniform_noise = tf.random_uniform([batch_size, self.noise_dim], minval=-1.0, maxval=1.0)

        normal_noise = tf.random_normal([batch_size, self.noise_dim])

        noise = normal_noise

        if self.num_class == 0:
            return noise

        else:
            if default_labels is None:
                default_labels = tf.random_uniform([batch_size], minval=0, maxval=self.num_class, dtype=tf.int32)
            self.labels = tf.placeholder_with_default(default_labels, [None])
            one_hot = tf.one_hot(self.labels, depth=self.num_class)
            return tf.concat([noise, one_hot], axis=1)


