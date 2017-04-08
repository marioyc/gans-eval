import matplotlib.pyplot as plt
import tensorflow as tf

from tqdm import tqdm

from common import build_model, sample_mixture_of_gaussians

def get_model_and_loss(params):
    data, data_score, z, samples, samples_score, discriminator_vars, generator_vars = build_model(params)
    discriminator_loss = -tf.reduce_mean(data_score - samples_score)
    generator_loss = -tf.reduce_mean(samples_score)
    return discriminator_vars, generator_vars, data, z, samples, discriminator_loss, generator_loss

def train(discriminator_vars, generator_vars, data, z, samples, discriminator_loss, generator_loss, dirname='gan'):
    sess = tf.Session()

    optimizer = tf.train.RMSPropOptimizer(learning_rate=5e-5)
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

    data_sampler = sample_mixture_of_gaussians(batch_size=params['batch_size'], **params['data'])
    z_sampler = tf.contrib.distributions.Normal(tf.zeros(params['z_dim']), tf.ones(params['z_dim'])).sample(params['batch_size'])

    clip_discriminator_vars_op = [var.assign(tf.clip_by_value(var, -0.01, 0.01))  for var in discriminator_vars]
    sess.run(tf.global_variables_initializer())

    visualization_step = 1000

    for i in tqdm(range(100000)):
        for j in range(5):
            data_batch, z_batch = sess.run([data_sampler, z_sampler])
            _, summary = sess.run([discriminator_train, summary_d_loss],
                            feed_dict={data: data_batch, z: z_batch})
            sess.run(clip_discriminator_vars_op)
        writer.add_summary(summary, i)

        z_batch = sess.run(z_sampler)
        _, summary = sess.run([generator_train, summary_g_loss],
                        feed_dict={z: z_batch})
        writer.add_summary(summary, i)

        if (i + 1) % visualization_step == 0:
            fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10,5))
            fig.suptitle(dirname)
            axes[0].set_title('Samples')
            axes[1].set_title('Data')

            for j in range(20):
                data_batch, z_batch = sess.run([data_sampler, z_sampler])
                x, y = sess.run([samples, data], feed_dict={data: data_batch, z: z_batch})
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
    }

    dirname = 'wgan'

    discriminator_vars, generator_vars, data, z, samples, discriminator_loss, generator_loss = get_model_and_loss(params)
    train(discriminator_vars, generator_vars, data, z, samples, discriminator_loss, generator_loss, dirname)
