import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.slim as slim

def sample_mixture_of_gaussians(n_mixture=8, std=0.01, radius=1):
    angles = np.linspace(0, 2 * np.pi, n_mixture + 1)[:-1]
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)

    cat = tf.contrib.distributions.Categorical(logits=tf.zeros(n_mixture))
    components = []
    for xi, yi in zip(x, y):
        components.append(tf.contrib.distributions.MultivariateNormalDiag([xi, yi], [std, std]))
    data = tf.contrib.distributions.Mixture(cat, components)
    return data

def discriminator(x, output_dim=1, n_layers=2, n_hidden=128,
        activation_fn=tf.nn.relu, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        h = slim.repeat(x, n_layers, slim.fully_connected, n_hidden,
                activation_fn=activation_fn,
                weights_initializer=layers.variance_scaling_initializer(
                    factor=2.0, mode='FAN_IN', uniform=True))
        logits = layers.fully_connected(h, output_dim, activation_fn=None,
                    weights_initializer=layers.variance_scaling_initializer(
                        factor=1.0, mode='FAN_AVG', uniform=True))
    return logits

def generator(z, output_dim=2, n_layers=2, n_hidden=128,
        activation_fn=tf.nn.relu):
    with tf.variable_scope('generator'):
        h = slim.repeat(z, n_layers, slim.fully_connected, n_hidden,
                activation_fn=activation_fn,
                weights_initializer=layers.variance_scaling_initializer(
                    factor=2.0, mode='FAN_IN', uniform=True))
        x = layers.fully_connected(h, output_dim, activation_fn=None,
                weights_initializer=layers.variance_scaling_initializer(
                    factor=1.0, mode='FAN_AVG', uniform=True))
    return x
