""" Charge la base de données MNIST"""


import torch

# Utilise automatiquement le GPU si CUDA est disponible
if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor


def train(nb_train, flatten=False):
    train = torch.load('data/train.pt')
    images, labels = train[0][:nb_train], train[1][:nb_train]
    images = images.type(dtype) / 255
    if flatten:
        images = images.view(len(images), -1)
    else:
        images = images.view(len(images), 1, 28, 28)
    return images, labels


def test(nb_val, flatten=False):
    train = torch.load('data/train.pt')
    images, labels = train[0][:nb_train], train[1][:nb_val]
    images = images.type(dtype) / 255
    if flatten:
        images = images.view(len(images), -1)
    else:
        images = images.view(len(images), 1, 28, 28)
    return images, labels
