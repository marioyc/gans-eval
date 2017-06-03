import matplotlib.pyplot as plt
import tensorflow as tf

from tqdm import tqdm

from gan import GAN
from common import build_model, sample_mixture_of_gaussians, discriminator

class WGAN(GAN):
    def __init__(self, params):
        self.params = params
        self.z_dim = params['z_dim']

        self.data_sampler = sample_mixture_of_gaussians(**params['data'])
        self.z_sampler = tf.contrib.distributions.Normal(tf.zeros(self.z_dim), tf.ones(self.z_dim))

        self.data = tf.placeholder(dtype=tf.float32, shape=[None, 2])
        self.z = tf.placeholder(dtype=tf.float32, shape=[None, self.z_dim])
        self.data_score, self.samples, self.samples_score, self.discriminator_vars, self.generator_vars = build_model(self.data, self.z, params)

        self.discriminator_loss = -tf.reduce_mean(self.data_score - self.samples_score)
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

        self.generator_loss = -tf.reduce_mean(self.samples_score)

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

    def _alternating_optimization(self, session):
        for j in range(self.discriminator_steps):
            data, z = session.run([self.data_batch_sampler, self.z_batch_sampler])
            _, summary_d = session.run([self.discriminator_train_op, self.summary_d_loss],
                            feed_dict={self.data: data, self.z: z})

            #TODO: fix WGAN clipping
            if not self.gradient_penalty:
                session.run(clip_discriminator_vars_op)

        z = session.run(self.z_batch_sampler)
        _, summary_g = session.run([self.generator_train_op, self.summary_g_loss],
                        feed_dict={self.z: z})

        return summary_d, summary_g

if __name__ == '__main__':
    params = {
        'batch_size': 256,
        'z_dim': 2,
        'data': {
            'n_mixture': 8,
            'std': 0.01,
            'radius': 1,
        },
        'generator': {
            'n_layers': 3,
            'n_hidden': 512,
            'activation_fn': tf.nn.relu,
        },
        'discriminator': {
            'n_layers': 3,
            'n_hidden': 512,
            'activation_fn': tf.nn.relu,
        },
        'gradient_penalty': True,
        'lambda': 0.1,
    }

    wgan = WGAN(params)
    wgan.train(discriminator_steps=5)
