import tensorflow as tf

from gan import GAN
from wgan import WGAN

params = {
    'z_dim': 64,
    'data': {
        'n_mixture': 8,
        'std': 0.1,
        'radius': 1,
    },
    'generator': {
        'n_layers': 4,
        'n_hidden': 256,
        'activation_fn': tf.nn.relu,
    },
    'discriminator': {
        'n_layers': 4,
        'n_hidden': 256,
        'activation_fn': tf.nn.relu,
    },
    'gradient_penalty': True,
    'lambda': 0.1,
}

wgan = WGAN(params)
wgan.train(iterations=10000, discriminator_steps=5, batch_size=256, visualization_step=2500)

params = {
    'z_dim': 64,
    'data': {
        'n_mixture': 8,
        'std': 0.1,
        'radius': 1,
    },
    'generator': {
        'n_layers': 4,
        'n_hidden': 256,
        'activation_fn': tf.nn.relu,
    },
    'discriminator': {
        'n_layers': 4,
        'n_hidden': 256,
        'activation_fn': tf.nn.relu,
    },
    'modified_objective': True,
    'optimization': {
        'algorithm': 'consensus',
        'gamma': 10,
    },
}

tf.reset_default_graph()
gan = GAN(params)
gan.train(iterations=10000, batch_size=256, visualization_step=2500)
