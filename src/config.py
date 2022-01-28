import torch.nn as nn
from ray import tune


large_netG = {
    'encoder': [
        {
            'in_channels': 1,
            'out_channels': 64,
            'kernel_size': 3,
            'stride': 1,
            'padding': 1,
            'normalize': True,
            'activation_func': nn.ReLU()
        },
        {
            'in_channels': 64,
            'out_channels': 64,
            'kernel_size': 4,
            'stride': 2,
            'padding': 1,
            'normalize': True,
            'activation_func': nn.ReLU()
        },
        {
            'in_channels': 64,
            'out_channels': 128,
            'kernel_size': 4,
            'stride': 2,
            'padding': 1,
            'normalize': True,
            'activation_func': nn.ReLU()
        },
        {
            'in_channels': 128,
            'out_channels': 256,
            'kernel_size': 4,
            'stride': 2,
            'padding': 1,
            'normalize': True,
            'activation_func': nn.ReLU()
        },
        {
            'in_channels': 256,
            'out_channels': 512,
            'kernel_size': 4,
            'stride': 2,
            'padding': 1,
            'normalize': True,
            'activation_func': nn.ReLU()
        },
        {
            'in_channels': 512,
            'out_channels': 512,
            'kernel_size': 4,
            'stride': 2,
            'padding': 1,
            'normalize': True,
            'activation_func': nn.ReLU()
        },
        {
            'in_channels': 512,
            'out_channels': 512,
            'kernel_size': 4,
            'stride': 2,
            'padding': 1,
            'normalize': True,
            'activation_func': nn.ReLU()
        },
        {
            'in_channels': 512,
            'out_channels': 512,
            'kernel_size': 4,
            'stride': 2,
            'padding': 1,
            'normalize': True,
            'activation_func': nn.ReLU()
        },
    ],
    'decoder': [
        {
            'in_channels': 512,
            'out_channels': 512,
            'kernel_size': 4,
            'stride': 2,
            'padding': 1,
            'normalize': True,
            'activation_func': nn.ReLU()
        },
        {
            'in_channels': 1024,
            'out_channels': 512,
            'kernel_size': 4,
            'stride': 2,
            'padding': 1,
            'normalize': True,
            'activation_func': nn.ReLU()
        },
        {
            'in_channels': 1024,
            'out_channels': 512,
            'kernel_size': 4,
            'stride': 2,
            'padding': 1,
            'normalize': True,
            'activation_func': nn.ReLU()
        },
        {
            'in_channels': 1024,
            'out_channels': 256,
            'kernel_size': 4,
            'stride': 2,
            'padding': 1,
            'normalize': True,
            'activation_func': nn.ReLU()
        },
        {
            'in_channels': 512,
            'out_channels': 128,
            'kernel_size': 4,
            'stride': 2,
            'padding': 1,
            'normalize': True,
            'activation_func': nn.ReLU()
        },
        {
            'in_channels': 256,
            'out_channels': 64,
            'kernel_size': 4,
            'stride': 2,
            'padding': 1,
            'normalize': True,
            'activation_func': nn.ReLU()
        },
        {
            'in_channels': 128,
            'out_channels': 64,
            'kernel_size': 4,
            'stride': 2,
            'padding': 1,
            'normalize': True,
            'activation_func': nn.ReLU()
        },
        {
            'in_channels': 128,
            'out_channels': 2,
            'kernel_size': 3,
            'stride': 1,
            'padding': 1,
            'normalize': True,
            'activation_func': nn.ReLU()
        },
        {
            'in_channels': 2,
            'out_channels': 2,
            'kernel_size': 3,
            'stride': 1,
            'padding': 1,
            'normalize': False,
            'activation_func': nn.Tanh()
        },
    ],
}


large_netD = {
    'blocks': [
        {
            'in_channels': 2,
            'out_channels': 64,
            'kernel_size': 4,
            'stride': 2,
            'padding': 1,
            'normalize': False,
            'activation_func': nn.LeakyReLU(0.2)
        },
        {
            'in_channels': 64,
            'out_channels': 128,
            'kernel_size': 4,
            'stride': 2,
            'padding': 1,
            'normalize': True,
            'activation_func': nn.LeakyReLU(0.2)
        },
        {
            'in_channels': 128,
            'out_channels': 256,
            'kernel_size': 4,
            'stride': 2,
            'padding': 1,
            'normalize': True,
            'activation_func': nn.LeakyReLU(0.2)
        },
        {
            'in_channels': 256,
            'out_channels': 512,
            'kernel_size': 3,
            'stride': 1,
            'padding': 1,
            'normalize': True,
            'activation_func': nn.LeakyReLU(0.2)
        },
        {
            'in_channels': 512,
            'out_channels': 1024,
            'kernel_size': 3,
            'stride': 1,
            'padding': 1,
            'normalize': True,
            'activation_func': nn.LeakyReLU(0.2)
        },
        {
            'in_channels': 1024,
            'out_channels': 1024,
            'kernel_size': 1,
            'stride': 1,
            'padding': 1,
            'normalize': True,
            'activation_func': nn.LeakyReLU(0.2)
        },
        {
            'in_channels': 1024,
            'out_channels': 1,
            'kernel_size': 1,
            'stride': 1,
            'padding': 0,
            'normalize': False,
            'activation_func': nn.Sigmoid()
        },
    ]
}


netGs = {
    'large_netG': large_netG
}


netDs = {
    'large_netD': large_netD
}


datasets = {
    'places365_20220124': {
        'src_dir': '/data/places365_standard',
        'dst_dir': '/data/places365_20220124',
        'n_classes': 3,
        'n_train_samples_per_class': 4000,
        'n_val_samples_per_class': 9,
    }
}


config = {
    'dataset': tune.grid_search(list(datasets.keys())),
    'netG': tune.grid_search(list(netGs.keys())),
    'netD': tune.grid_search(list(netDs.keys())),
    'lrG': tune.grid_search([2e-4]),
    'lrD': tune.grid_search([2e-4]),
    'p': tune.grid_search([0.7]),
    'lambda': tune.grid_search([30, 100, 300]),
    'batch_size': 32,
}
