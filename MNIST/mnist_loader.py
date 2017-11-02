""" Charge la base de données MNIST"""


import torch

# Utilise automatiquement le GPU si CUDA est disponible
if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor


def train(nb_train, flatten=False):
    images = torch.load('data/train_images.pt')[:nb_train]
    labels = torch.load('data/train_labels.pt')[:nb_train]
    images = images.type(dtype) / 255
    if flatten:
        images = images.view(len(images), -1)
    else:
        images = images.view(len(images), 1, 28, 28)
    return images, labels


def test(nb_val, flatten=False):
    images = torch.load('data/test_images.pt')[:nb_val]
    labels = torch.load('data/test_labels.pt')[:nb_val]
    images = images.type(dtype) / 255
    if flatten:
        images = images.view(len(images), -1)
    else:
        images = images.view(len(images), 1, 28, 28)
    return images, labels
