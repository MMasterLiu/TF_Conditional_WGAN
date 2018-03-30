import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def MNIST(batch_size):
    assert type(batch_size) is int

    class Gen:
        def __init__(self):
            mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
            self.images = mnist.train.images
            self.labels = mnist.train.labels
            self.order = np.arange(len(self.images))
        def __call__(self):
            np.random.shuffle(self.order)
            for o in self.order:
                i = self.images[o].reshape([28, 28])
                i = np.pad(i, [2,2], mode="constant") # [32,32]
                i = i.reshape([32,32, 1]) # [32,32, 1]
                i = i * 2.0 - 1.0 # [-1.0, 1.0)
                l = self.labels[o]
                yield (i, l)
    gen = Gen()
    image_dataset = tf.data.Dataset.from_generator(gen, output_types=(tf.float32, tf.int32), output_shapes=(tf.TensorShape([32,32,1]), tf.TensorShape([])))
    image_dataset = image_dataset.repeat()
    iterator = image_dataset.batch(batch_size).prefetch(8).make_one_shot_iterator()
    batched_images_and_labels = iterator.get_next()
    return batched_images_and_labels
