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

        self._init_optimization()

    def _init_optimization(self):
        self._create_optimizers()
        self._create_training_ops()
        self.summary_d_loss = tf.summary.scalar('discriminator_loss', self.discriminator_loss)
        self.summary_g_loss = tf.summary.scalar('generator_loss', self.generator_loss)

    def _create_optimizers(self):
        if self.params['optimization']['algorithm'] == 'consensus':
            self.optimizer = tf.train.RMSPropOptimizer(1e-4)
        elif self.params['optimization']['algorithm'] == 'alternating':
            self.discriminator_optimizer = tf.train.AdamOptimizer()
            self.generator_optimizer = tf.train.AdamOptimizer()

    def _create_training_ops(self):
        if self.params['optimization']['algorithm'] == 'consensus':
            self.name += ' concensus optimization'

            gamma = self.params['optimization']['gamma']
            discriminator_gradients = tf.gradients(self.discriminator_loss, self.discriminator_vars)
            generator_gradients = tf.gradients(self.generator_loss, self.generator_vars)
            gradients = discriminator_gradients + generator_gradients
            variables = self.discriminator_vars + self.generator_vars

            L = 0.5 * sum([tf.reduce_sum(tf.square(g)) for g in gradients])
            Jgrads = tf.gradients(L, gradients)

            gradients_to_apply = [(g + gamma * Lg, v) for g, Lg, v in zip(gradients, Jgrads, variables)]

            with tf.control_dependencies([g for (g, v) in gradients_to_apply]):
                self.train_op = self.optimizer.apply_gradients(gradients_to_apply)
        elif self.params['optimization']['algorithm'] == 'alternating':
            self.name += ' alternating optimization'

            self.discriminator_steps = self.params['optimization']['discriminator_steps']
            self.discriminator_train_op = self.discriminator_optimizer.minimize(
                            self.discriminator_loss, var_list=self.discriminator_vars)
            self.generator_train_op = self.generator_optimizer.minimize(
                            self.generator_loss, var_list=self.generator_vars)

    def _optimization_step(self):
        session = tf.get_default_session()

        if self.params['optimization']['algorithm'] == 'consensus':
            return self._consensus_optimization(session)
        elif self.params['optimization']['algorithm'] == 'alternating':
            return self._alternating_optimization(session)

    def _consensus_optimization(self, session):
        data, z = session.run([self.data_batch_sampler, self.z_batch_sampler])

        _, summary_d, summary_g = session.run([self.train_op,
                                    self.summary_d_loss, self.summary_g_loss],
                                    feed_dict={self.data: data, self.z: z})
        return summary_d, summary_g

    def _alternating_optimization(self, session):
        for i in range(self.discriminator_steps):
            data, z = session.run([self.data_batch_sampler, self.z_batch_sampler])
            _, summary_d = session.run([self.discriminator_train_op, self.summary_d_loss],
                            feed_dict={self.data: data, self.z: z})

        z = session.run(self.z_batch_sampler)
        _, summary_g = session.run([self.generator_train_op, self.summary_g_loss],
                        feed_dict={self.z: z})

        return summary_d, summary_g

    def _visualization_2d(self, step, visualization_batches, batch_size, path=None, plot=False):
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

        if path is not None:
            fig.savefig(path + '/{0:06d}.png'.format(step + 1))
        if plot:
            plt.show()
        plt.close(fig)

    def train(self, iterations=10000, batch_size=64, save_step=1000,
            visualization_batches=10, dirname=None):
        session = tf.Session()

        with session.as_default():
            self.data_batch_sampler = self.data_sampler.sample(batch_size)
            self.z_batch_sampler = self.z_sampler.sample(batch_size)

            if dirname is None:
                dirname = '-'.join(self.name.lower().split())
            logs_path = 'logs/{}'.format(dirname)
            output_path = 'output/{}'.format(dirname)

            if tf.gfile.Exists(logs_path):
                tf.gfile.DeleteRecursively(logs_path)
            self.writer = tf.summary.FileWriter(logs_path, session.graph)

            if tf.gfile.Exists(output_path):
                tf.gfile.DeleteRecursively(output_path)
            tf.gfile.MakeDirs(output_path)

            session.run(tf.global_variables_initializer())

            for i in tqdm(range(iterations)):
                summary_d, summary_g = self._optimization_step()
                self.writer.add_summary(summary_d, i)
                self.writer.add_summary(summary_g, i)

                if (i + 1) % save_step == 0:
                    self._visualization_2d(i, visualization_batches, batch_size, path=output_path)

            self._visualization_2d(i, visualization_batches, batch_size, plot=True)

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
