import matplotlib.pyplot as plt
import tensorflow as tf

from tqdm import tqdm

from common import build_model, sample_mixture_of_gaussians, discriminator

class WGAN:
    def __init__(self, params):
        self.data_params = params['data']
        self.z_dim = params['z_dim']

        self.data = tf.placeholder(dtype=tf.float32, shape=[None, 2])
        self.z = tf.placeholder(dtype=tf.float32, shape=[None, self.z_dim])
        self.data_score, self.samples, self.samples_score, self.discriminator_vars, self.generator_vars = build_model(self.data, self.z, params)

        self.discriminator_loss = -tf.reduce_mean(self.data_score - self.samples_score)
        self.gradient_penalty = params['gradient_penalty']
        if params['gradient_penalty']:
            e = tf.contrib.distributions.Uniform().sample([64, 1])
            x = e * self.data + (1 - e) * self.samples
            x_score = discriminator(x, **params['discriminator'], reuse=True)
            gradients = tf.gradients(x_score, [x])[0]
            gradients_l2 = tf.sqrt(tf.reduce_sum(gradients ** 2, axis=1))
            gradient_penalty = tf.reduce_mean((gradients_l2 - 1) ** 2)
            self.discriminator_loss += 10 * gradient_penalty

        self.generator_loss = -tf.reduce_mean(self.samples_score)

    def train(self, iterations=100000, discriminator_steps=5, batch_size=64,
            visualization_step=1000, visualization_batches=20, dirname='wgan'):
        sess = tf.Session()

        if self.gradient_penalty:
            discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9)
            generator_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9)
        else:
            discriminator_optimizer = tf.train.RMSPropOptimizer(learning_rate=5e-5)
            generator_optimizer = tf.train.RMSPropOptimizer(learning_rate=5e-5)

        discriminator_train = discriminator_optimizer.minimize(self.discriminator_loss, var_list=self.discriminator_vars)
        generator_train = generator_optimizer.minimize(self.generator_loss, var_list=self.generator_vars)

        summary_d_loss = tf.summary.scalar('discriminator_loss', self.discriminator_loss)
        summary_g_loss = tf.summary.scalar('generator_loss', self.generator_loss)

        logs_path = 'logs/{}'.format(dirname)
        output_path = 'output/{}'.format(dirname)

        if tf.gfile.Exists(logs_path):
            tf.gfile.DeleteRecursively(logs_path)
        if tf.gfile.Exists(output_path):
            tf.gfile.DeleteRecursively(output_path)
        tf.gfile.MakeDirs(output_path)

        writer = tf.summary.FileWriter(logs_path, sess.graph)

        data_sampler = sample_mixture_of_gaussians(batch_size=batch_size, **self.data_params)
        z_sampler = tf.contrib.distributions.Normal(tf.zeros(self.z_dim), tf.ones(self.z_dim)).sample(batch_size)

        if not self.gradient_penalty:
            clip_discriminator_vars_op = [var.assign(tf.clip_by_value(var, -0.01, 0.01))  for var in discriminator_vars]

        sess.run(tf.global_variables_initializer())

        for i in tqdm(range(iterations)):
            for j in range(discriminator_steps):
                data, z = sess.run([data_sampler, z_sampler])
                _, summary = sess.run([discriminator_train, summary_d_loss],
                                feed_dict={self.data: data, self.z: z})

                if not self.gradient_penalty:
                    sess.run(clip_discriminator_vars_op)
            writer.add_summary(summary, i)

            z = sess.run(z_sampler)
            _, summary = sess.run([generator_train, summary_g_loss],
                            feed_dict={self.z: z})
            writer.add_summary(summary, i)

            if (i + 1) % visualization_step == 0:
                fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10,5))
                fig.suptitle(dirname)
                axes[0].set_title('Samples')
                axes[1].set_title('Data')

                for j in range(visualization_batches):
                    data, z = sess.run([data_sampler, z_sampler])
                    x, y = sess.run([self.samples, self.data],
                            feed_dict={self.data: data, self.z: z})
                    axes[0].scatter(x[:, 0], x[:, 1], c='blue', edgecolor='none')
                    axes[1].scatter(y[:, 0], y[:, 1], c='green', edgecolor='none')

                fig.savefig(output_path + '/{0:06d}.png'.format(i + 1))
                plt.close(fig)

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
        'gradient_penalty': True,
    }

    dirname = 'wgan'
    if params['gradient_penalty']:
        dirname = 'wgan-gradient-penalty'

    wgan = WGAN(params)
    wgan.train(dirname=dirname)
