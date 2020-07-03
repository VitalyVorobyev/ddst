#! /usr/bin/env python

import tensorflow as tf
from tensorflow import keras as k

""" Train NN to learn 3D or 5D pdf """

def umaxlh_loss(model, input_, norm_sample):
    """ loss function for the unbinned maximum likelihood fit """
    return -2.*tf.reduce_sum(
        tf.math.log(
            model(input_) / tf.reduce_sum(model(norm_sample))
        )
    )


def uniform_sample(space, n):
    """ Generates uniform sample in a rectangular space """
    return tf.stack([
        tf.random.uniform([n], lo, hi) for lo, hi in space
    ], axis=1)


def model(input_, space, norm_size):
    """ """
    # Create an optimizer.

    model = k.Sequential(
        k.layers.Dense(64, activation='relu', input_shape=(len(space))),
        k.layers.Dense(32, activation='relu'),
        k.layers.Dense(1, activation='sigmoid')
    )

    loss_fn = lambda: umaxlh_loss(model, input_, uniform_sample(space, norm_size))
    var_list_fn = lambda: model.trainable_weights

    # opt = k.optimizers.SGD(learning_rate=0.1)
    # loss_fn = lambda: tf.keras.losses.mse(model(input), output)
    # var_list_fn = lambda: model.trainable_weights
    # for input, output in data:
    #     opt.minimize(loss_fn, var_list_fn)
