import os
import PIL.Image
import numpy as np
import tensorflow as tf

import src.model as Model
import src.mnist as MNIST

# hyperparameters
conditioned_model = True # immutable. only boolean type works. (i.e., cannot use tensor type.)
same_labels = True # use same labels as real_labels for fake_labels. in my few experiments, this sometimes boosted training speed, but cannot say strongly.
batch_size = 32
large_batch_size = 128
start_using_large_minibatch = 5000 # 2nd-stage training will start from this iteration.
num_disc_overtraining = 5 # number of train_disc running per one train_gen running.
clip_disc_params = False # usually WGAN-GP doesn't need weight-clipping.

# constant values (not hyperparameters)
num_channel = 1
image_size = 32
num_class = 10 if conditioned_model else 0



print("load dataset")
use_supplement = tf.placeholder_with_default(False, [])
few_real_images, few_real_labels = MNIST.MNIST(batch_size)
supplementary_real_images, supplementary_real_labels = MNIST.MNIST(large_batch_size - batch_size)
many_real_images = tf.concat([few_real_images, supplementary_real_images], axis=0)
many_real_labels = tf.concat([few_real_labels, supplementary_real_labels], axis=0)
minibatch_real_images, minibatch_real_labels = tf.cond(use_supplement, lambda: (many_real_images, many_real_labels), lambda: (few_real_images, few_real_labels))

print("build model")
model = Model.Model(num_channel=num_channel, num_class=num_class)
minibatch_batch_size = tf.shape(minibatch_real_images)[0]
noise_batch_size = tf.placeholder_with_default(minibatch_batch_size, [])
default_fake_labels = minibatch_real_labels if conditioned_model and same_labels else None # None means random-labeled or non-conditioned.
noise = model._random_noise(noise_batch_size, default_labels=default_fake_labels)
real_labels = minibatch_real_labels if conditioned_model else None
model.build_network_outputs(noise, minibatch_real_images, fake_labels=default_fake_labels, real_labels=real_labels)
model.build_losses()

print("build opt")
opt = tf.train.AdamOptimizer(learning_rate=2.0e-4)
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="Generator")):
    gvs = opt.compute_gradients(model.gen_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Generator"))
    train_gen = opt.apply_gradients([(tf.clip_by_norm(grad, tf.sqrt(tf.cast(tf.reduce_prod(tf.shape(var)), tf.float32))), var) for grad, var in gvs if grad is not None])
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="Discriminator")):
    gvs = opt.compute_gradients(model.disc_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Discriminator"))
    train_disc = opt.apply_gradients([(tf.clip_by_norm(grad, tf.sqrt(tf.cast(tf.reduce_prod(tf.shape(var)), tf.float32))), var) for grad, var in gvs if grad is not None])
if clip_disc_params:
    clip_disc = [param.assign(tf.clip_by_value(param, -0.1, 0.1)) for param in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Discriminator") if  "kernel" in param.name]
else:
    clip_disc = tf.no_op()

sess = tf.InteractiveSession()
print("initialize")
sess.run(tf.global_variables_initializer())

def printout(dirname=None, suffix=None):
    if dirname is None: dirname = "."
    if suffix is None: suffix = ""
    if conditioned_model:
        labels = [i for _ in range(10) for i in range(10)]
        gimg = sess.run(model.gen_image, feed_dict={noise_batch_size:100, model.labels:labels}) # [batch_size, 32,32, 3]
    else:
        gimg = sess.run(model.gen_image, feed_dict={noise_batch_size:100}) # [batch_size, 32,32, 3]
    gimg = gimg.reshape([10, 10, model.image_size,model.image_size, model.num_channel]).transpose([0,2, 1,3, 4])
    gimg = gimg.reshape([gimg.shape[0]*gimg.shape[1], gimg.shape[2]*gimg.shape[3], model.num_channel]).squeeze()
    gimg = ((gimg * 128.0) + 128.0).clip(0.0, 255.0).astype("uint8")
    img = PIL.Image.fromarray(gimg)
    fname = "sample{}.png".format(suffix)
    img.save(os.path.join(dirname, fname))

def run(feed_use_supplement, start=0, end=100000):
    if feed_use_supplement:
        train_feed_dict = {use_supplement:feed_use_supplement}
    else:
        train_feed_dict = None

    for iteration in range(start, end):
        print(iteration)
        wd, _ = sess.run([model.w_distance, train_gen], feed_dict=train_feed_dict)
        print(wd)
        wd = -1
        while wd < 0: # since we expect discriminator (critic) approximates w-distance well, and w-distance is a distance, let's wait until it satisfies non-negativity at least for some minibatch.
            for _ in range(num_disc_overtraining):
                wd, _ = sess.run([model.w_distance, train_disc], feed_dict=train_feed_dict)
                sess.run(clip_disc)
            print(wd)
        if iteration % 100 == 0:
            dirname = "sample"
            tf.gfile.MakeDirs(dirname)
            printout(dirname=dirname, suffix="{0:08}".format(iteration))

print("start training")
run(feed_use_supplement=False, end=start_using_large_minibatch)
print("start 2nd-stage training")
run(feed_use_supplement=True, start=start_using_large_minibatch, end=100000)
