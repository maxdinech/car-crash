""" Charge la base de données MNIST"""


import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import os
import shutil

# Utilise automatiquement le GPU si CUDA est disponible
if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor

def créer_MNIST():
    _ = dsets.MNIST(root='data/',
                    train=True, 
                    transform=transforms.ToTensor(),
                    download=True)
    shutil.move('data/processed/training.pt', 'data/train.pt')
    shutil.move('data/processed/test.pt', 'data/test.pt')
    os.remove('data/raw')
    os.remove('data/processed')

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
    images, labels = train[0][:nb_val], train[1][:nb_val]
    images = images.type(dtype) / 255
    if flatten:
        images = images.view(len(images), -1)
    else:
        images = images.view(len(images), 1, 28, 28)
    return images, labels
