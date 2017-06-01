import matplotlib.pyplot as plt
import tensorflow as tf

from tqdm import tqdm

from common import build_model, sample_mixture_of_gaussians

class GAN:
    def __init__(self, params):
        self.params = params
        self.z_dim = params['z_dim']

        self.data_sampler = sample_mixture_of_gaussians(**params['data'])
        self.z_sampler = tf.contrib.distributions.Normal(tf.zeros(self.z_dim), tf.ones(self.z_dim))

        self.data = tf.placeholder(dtype=tf.float32, shape=[None, 2])
        self.z = tf.placeholder(dtype=tf.float32, shape=[None, self.z_dim])
        self.data_score, self.samples, self.samples_score, self.discriminator_vars, self.generator_vars = build_model(self.data, self.z, params)

        self.discriminator_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.data_score), logits=self.data_score)
            + tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.samples_score), logits=self.samples_score))

        if params['modified_objective']:
            self.modified_objective = True
            self.name = 'GAN modified objective'
            self.generator_loss = -tf.reduce_mean(tf.nn.sigmoid(self.samples_score))
        else:
            self.modified_objective = False
            self.name = 'GAN'
            self.generator_loss = -tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.samples_score), logits=self.samples_score))

        if params['optimization']['algorithm'] == 'consensus':
            self.name += ' and concensus optimization'
            gamma = params['optimization']['gamma']
            L = sum([tf.reduce_sum(tf.square(g)) / 2 for g in tf.gradients(self.discriminator_loss, self.discriminator_vars)]) \
                + sum([tf.reduce_sum(tf.square(g)) / 2 for g in tf.gradients(self.generator_loss, self.generator_vars)])
            self.discriminator_loss += gamma * L
            self.generator_loss += gamma * L

    def _visualization_2d(self, step, visualization_batches, batch_size, path):
        session = tf.get_default_session()
        fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10,5))
        fig.suptitle(self.name)
        axes[0].set_title('Samples')
        axes[1].set_title('Data')

        for j in range(visualization_batches):
            data, z = session.run([self.data_batch_sampler, self.z_batch_sampler])
            x, y = session.run([self.samples, self.data],
                    feed_dict={self.data: data, self.z: z})
            axes[0].scatter(x[:, 0], x[:, 1], c='blue', edgecolor='none')
            axes[1].scatter(y[:, 0], y[:, 1], c='green', edgecolor='none')

        fig.savefig(path + '/{0:06d}.png'.format(step + 1))
        plt.close(fig)

    def _get_optimizers(self):
        if self.params['optimization']['algorithm'] == 'consensus':
            discriminator_optimizer = tf.train.RMSPropOptimizer(1e-4)
            generator_optimizer = tf.train.RMSPropOptimizer(1e-4)
        else:
            discriminator_optimizer = tf.train.AdamOptimizer()
            generator_optimizer = tf.train.AdamOptimizer()
        return discriminator_optimizer, generator_optimizer

    def _optimization_step(self):
        session = tf.get_default_session()

        if self.params['optimization']['algorithm'] == 'consensus':
            data, z = session.run([self.data_batch_sampler, self.z_batch_sampler])

            calculated_d_grad = session.run(self.discriminator_gradients,
                                feed_dict={self.data: data, self.z: z})
            calculated_g_grad = session.run(self.generator_gradients,
                                feed_dict={self.data: data, self.z: z})

            d_feed_dict = {self.data: data, self.z: z}
            for entry, calculated_entry in zip(self.d_grad, calculated_d_grad):
                d_feed_dict[entry[0]] = calculated_entry[0]

            g_feed_dict = {self.data: data, self.z: z}
            for entry, calculated_entry in zip(self.g_grad, calculated_g_grad):
                g_feed_dict[entry[0]] = calculated_entry[0]

            _, summary_d = session.run([self.discriminator_train, self.summary_d_loss],
                                feed_dict=d_feed_dict)
            _, summary_g = session.run([self.generator_train, self.summary_g_loss],
                                feed_dict=g_feed_dict)
        else:
            for i in range(self.discriminator_steps):
                data, z = session.run([self.data_batch_sampler, self.z_batch_sampler])
                _, summary_d = session.run([self.discriminator_train, self.summary_d_loss],
                                feed_dict={self.data: data, self.z: z})

            z = session.run(self.z_batch_sampler)
            _, summary_g = session.run([self.generator_train, self.summary_g_loss],
                            feed_dict={self.z: z})

        return summary_d, summary_g

    def train(self, iterations=100000, discriminator_steps=1, batch_size=64,
            visualization_step=1000, visualization_batches=20, dirname=None):
        session = tf.Session()

        with session.as_default():
            self.data_batch_sampler = self.data_sampler.sample(batch_size)
            self.z_batch_sampler = self.z_sampler.sample(batch_size)

            self.discriminator_optimizer, self.generator_optimizer = self._get_optimizers()

            if 'optimization' in self.params and self.params['optimization']['algorithm'] == 'consensus':
                self.discriminator_gradients = self.discriminator_optimizer.compute_gradients(self.discriminator_loss, var_list=self.discriminator_vars)
                self.generator_gradients = self.generator_optimizer.compute_gradients(self.generator_loss, var_list=self.generator_vars)

                self.d_grad = [(tf.placeholder(tf.float32), v) for v in self.discriminator_vars]
                self.g_grad = [(tf.placeholder(tf.float32), v) for v in self.generator_vars]

                self.discriminator_train = self.discriminator_optimizer.apply_gradients(self.d_grad)
                self.generator_train = self.generator_optimizer.apply_gradients(self.g_grad)
            else:
                self.discriminator_train = self.discriminator_optimizer.minimize(
                                self.discriminator_loss, var_list=self.discriminator_vars)
                self.discriminator_steps = discriminator_steps
                self.generator_train = self.generator_optimizer.minimize(
                                self.generator_loss, var_list=self.generator_vars)

            if dirname is None:
                dirname = '-'.join(self.name.lower().split())
            logs_path = 'logs/{}'.format(dirname)
            output_path = 'output/{}'.format(dirname)

            if tf.gfile.Exists(logs_path):
                tf.gfile.DeleteRecursively(logs_path)
            if tf.gfile.Exists(output_path):
                tf.gfile.DeleteRecursively(output_path)
            tf.gfile.MakeDirs(output_path)

            self.summary_d_loss = tf.summary.scalar('discriminator_loss', self.discriminator_loss)
            self.summary_g_loss = tf.summary.scalar('generator_loss', self.generator_loss)
            self.writer = tf.summary.FileWriter(logs_path, session.graph)

            session.run(tf.global_variables_initializer())

            for i in tqdm(range(iterations)):
                summary_d, summary_g = self._optimization_step()
                self.writer.add_summary(summary_d, i)
                self.writer.add_summary(summary_g, i)

                if (i + 1) % visualization_step == 0:
                    self._visualization_2d(i, visualization_batches, batch_size, output_path)

        session.close()

if __name__ == '__main__':
    params = {
        'batch_size': 64,
        'z_dim': 10,
        'data': {
            'n_mixture': 8,
            'std': 0.01,
            'radius': 1,
        },
        'generator': {
            'n_layers': 3,
            'n_hidden': 128,
            'activation_fn': tf.nn.relu,
        },
        'discriminator': {
            'n_layers': 2,
            'n_hidden': 128,
            'activation_fn': tf.nn.relu,
        },
        'modified_objective': True,
    }

    gan = GAN(params)
    gan.train()
