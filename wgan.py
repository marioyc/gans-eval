import tensorflow as tf

from gan import GAN
from common import sample_mixture_of_gaussians, discriminator, generator

class WGAN(GAN):
    def __init__(self, params):
        self.params = params
        self.z_dim = params['z_dim']

        data_sampler = sample_mixture_of_gaussians(**params['data'])
        z_sampler = tf.contrib.distributions.Normal(tf.zeros(self.z_dim), tf.ones(self.z_dim))
        self.batch_size = tf.placeholder(tf.int32, shape=())

        self.data = data_sampler.sample(self.batch_size)
        data_score = discriminator(self.data, **params['discriminator'])

        self.z = z_sampler.sample(self.batch_size)
        self.samples = generator(self.z, **params['generator'])
        samples_score = discriminator(self.samples, **params['discriminator'], reuse=True)

        self.discriminator_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
        self.generator_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')

        self.discriminator_loss = -tf.reduce_mean(data_score - samples_score)
        if params['gradient_penalty']:
            self.gradient_penalty = True
            self.name = 'WGAN gradient penalty'
            e = tf.contrib.distributions.Uniform().sample([tf.shape(self.data)[0], 1])
            x = e * self.data + (1 - e) * self.samples
            x_score = discriminator(x, **params['discriminator'], reuse=True)
            gradients = tf.gradients(x_score, [x])[0]
            gradients_l2 = tf.sqrt(tf.reduce_sum(gradients ** 2, axis=1))
            gradient_penalty = tf.reduce_mean((gradients_l2 - 1) ** 2)
            self.discriminator_loss += params['lambda'] * gradient_penalty
        else:
            self.gradient_penalty = False
            self.name = 'WGAN'

        self.generator_loss = -tf.reduce_mean(samples_score)

        self._init_optimization()

    def _create_optimizers(self):
        if self.params['optimization']['algorithm'] == 'consensus':
            self.optimizer = tf.train.RMSPropOptimizer(1e-4)
        elif self.params['optimization']['algorithm'] == 'alternating':
            if self.gradient_penalty:
                self.discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9)
                self.generator_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9)
            else:
                self.discriminator_optimizer = tf.train.RMSPropOptimizer(learning_rate=5e-5)
                self.generator_optimizer = tf.train.RMSPropOptimizer(learning_rate=5e-5)

        if not self.gradient_penalty:
            self.clip_discriminator_op = [var.assign(tf.clip_by_value(var, -0.01, 0.01))  for var in self.discriminator_vars]

    def _consensus_optimization(self, session, batch_size):
        _, summary_d, summary_g = session.run([self.train_op,
                                    self.summary_d, self.summary_g],
                                    feed_dict={self.batch_size: batch_size})

        if not self.gradient_penalty:
            session.run(self.clip_discriminator_op)

        return summary_d, summary_g

    def _alternating_optimization(self, session, batch_size):
        for j in range(self.discriminator_steps):
            _, summary_d = session.run([self.discriminator_train_op, self.summary_d],
                            feed_dict={self.batch_size: batch_size})

            if not self.gradient_penalty:
                session.run(self.clip_discriminator_op)

        _, summary_g = session.run([self.generator_train_op, self.summary_g],
                        feed_dict={self.batch_size: batch_size})

        return summary_d, summary_g
