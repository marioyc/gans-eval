import matplotlib.pyplot as plt
import tensorflow as tf

from tqdm import tqdm

from common import build_model, sample_mixture_of_gaussians

class GAN:
    def __init__(self, params):
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
        discriminator_optimizer = tf.train.AdamOptimizer()
        generator_optimizer = tf.train.AdamOptimizer()
        return discriminator_optimizer, generator_optimizer

    def _optimization_step(self):
        session = tf.get_default_session()

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
