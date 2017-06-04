import tensorflow as tf

from gan import GAN
from common import sample_mixture_of_gaussians, discriminator, generator

class CramerGAN(GAN):
    def __init__(self, params):
        self.name = 'Cramer GAN'
        self.params = params
        self.z_dim = params['z_dim']

        data_sampler = sample_mixture_of_gaussians(**params['data'])
        z_sampler = tf.contrib.distributions.Normal(tf.zeros(self.z_dim), tf.ones(self.z_dim))
        self.batch_size = tf.placeholder(tf.int32, shape=())

        self.data = data_sampler.sample(self.batch_size)
        data_h = discriminator(self.data, **params['discriminator'])

        self.z = z_sampler.sample(self.batch_size)
        self.samples = generator(self.z, **params['generator'])
        samples_h = discriminator(self.samples, **params['discriminator'], reuse=True)

        self.z2 = z_sampler.sample(self.batch_size)
        self.samples2 = generator(self.z2, **params['generator'])
        samples2_h = discriminator(self.samples2, **params['discriminator'], reuse=True)

        self.discriminator_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
        self.generator_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')

        f = lambda h: tf.sqrt(tf.reduce_sum((h - samples2_h) ** 2, axis=1)) \
                        - tf.sqrt(tf.reduce_sum(h ** 2, axis=1))
        self.discriminator_loss = tf.reduce_mean(f(data_h) - f(samples_h))

        e = tf.contrib.distributions.Uniform().sample([self.batch_size, 1])
        x = e * self.data + (1 - e) * self.samples
        x_h = discriminator(x, **params['discriminator'], reuse=True)
        gradients = tf.gradients(f(x_h), [x])[0]
        gradients_l2 = tf.sqrt(tf.reduce_sum(gradients ** 2, axis=1))
        gradient_penalty = tf.reduce_mean((gradients_l2 - 1) ** 2)
        self.discriminator_loss += params['lambda'] * gradient_penalty

        g = lambda h1, h2: tf.sqrt(tf.reduce_sum((h1 - h2) ** 2, axis=1))
        self.generator_loss = tf.reduce_mean(g(data_h, samples_h) \
                                + g(data_h, samples2_h) \
                                - g(samples_h, samples2_h))

        self._init_optimization()

    def _create_optimizers(self):
        if self.params['optimization']['algorithm'] == 'consensus':
            self.optimizer = tf.train.RMSPropOptimizer(1e-4)
        elif self.params['optimization']['algorithm'] == 'alternating':
            self.discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9)
            self.generator_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9)
