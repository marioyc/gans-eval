import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from tqdm import tqdm

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

def build_model(params):
    data = sample_mixture_of_gaussians(params['batch_size'], params['n_mixture'],
                params['std'], params['radius'])
    data_score = discriminator(data, **params['discriminator'])

    z = tf.contrib.distributions.Normal(tf.zeros(params['z_dim']), tf.ones(params['z_dim']))
    z = z.sample(params['batch_size'])
    samples = generator(z, **params['generator'])
    samples_score = discriminator(samples, **params['discriminator'], reuse=True)

    discriminator_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
    generator_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')

    discriminator_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(data_score), logits=data_score)
        + tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(samples_score), logits=samples_score))

    if params['modified_objective']:
        generator_loss = -tf.reduce_mean(tf.nn.sigmoid(samples_score))
    else:
        generator_loss = -tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(samples_score), logits=samples_score))

    return discriminator_vars, generator_vars, data, samples, discriminator_loss, generator_loss

def train(discriminator_vars, generator_vars, data, samples, discriminator_loss, generator_loss, dirname='gan'):
    sess = tf.Session()

    optimizer = tf.train.AdamOptimizer()
    discriminator_train = optimizer.minimize(discriminator_loss, var_list=discriminator_vars)
    generator_train = optimizer.minimize(generator_loss, var_list=generator_vars)

    summary_d_loss = tf.summary.scalar('discriminator_loss', discriminator_loss)
    summary_g_loss = tf.summary.scalar('generator_loss', generator_loss)

    logs_path = 'logs/{}'.format(dirname)
    output_path = 'output/{}'.format(dirname)

    if tf.gfile.Exists(logs_path):
        tf.gfile.DeleteRecursively(logs_path)
    if tf.gfile.Exists(output_path):
        tf.gfile.DeleteRecursively(output_path)
    tf.gfile.MakeDirs(output_path)

    writer = tf.summary.FileWriter(logs_path, sess.graph)

    sess.run(tf.global_variables_initializer())

    visualization_step = 1000

    for i in tqdm(range(20000)):
        _, summary = sess.run([discriminator_train, summary_d_loss])
        writer.add_summary(summary, i)
        _, summary = sess.run([generator_train, summary_g_loss])
        writer.add_summary(summary, i)

        if (i + 1) % visualization_step == 0:
            x, y = sess.run([samples, data])
            fig = plt.figure(figsize=(5,5))
            plt.scatter(x[:, 0], x[:, 1], edgecolor='none')
            plt.scatter(y[:, 0], y[:, 1], c='g', edgecolor='none')
            plt.axis('off')
            fig.savefig(output_path + '/{}.png'.format(i + 1), bbox_inches='tight')
            plt.close(fig)

if __name__ == '__main__':
    params = {
        'n_mixture': 8,
        'batch_size': 512,
        'z_dim': 10,
        'std': 0.01,
        'radius': 1,
        'generator': {
            'n_layers': 3,
            'n_hidden': 128,
        },
        'discriminator': {
            'n_layers': 2,
            'n_hidden': 128,
        },
        'modified_objective': True,
    }

    dirname = 'gan'
    if params['modified_objective']:
        dirname = 'modified-gan'

    discriminator_vars, generator_vars, data, samples, discriminator_loss, generator_loss = build_model(params)
    train(discriminator_vars, generator_vars, data, samples, discriminator_loss, generator_loss, dirname)
