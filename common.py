import numpy as np

def sample_mixture_of_gaussians(batch_size=64, n_mixture=8, std=0.01, radius=1):
    angles = np.linspace(0, 2 * np.pi, n_mixture)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)

    cat = tf.contrib.distributions.Categorical(logits=tf.zeros(n_mixture))
    components = []
    for xi, yi in zip(x, y):
        components.append(tf.contrib.distributions.MultivariateNormalDiag([xi, yi], [std, std]))
    data = tf.contrib.distributions.Mixture(cat, components)
    return data.sample(batch_size)

def discriminator(x, n_layers=2, n_hidden=128, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        h = slim.repeat(x, n_layers, slim.fully_connected, n_hidden, activation_fn=tf.nn.tanh)
        logits = tf.contrib.layers.fully_connected(h, 1, activation_fn=None)
    return logits

def generator(z, n_layers=2, n_hidden=128):
    with tf.variable_scope('generator'):
        h = slim.repeat(z, n_layers, slim.fully_connected, n_hidden, activation_fn=tf.nn.tanh)
        x = tf.contrib.layers.fully_connected(h, 2, activation_fn=None)
    return x
