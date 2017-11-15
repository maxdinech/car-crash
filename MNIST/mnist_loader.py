"""
Charge la base de données MNIST

"""



import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import os.path
import shutil


# Utilise automatiquement le GPU si CUDA est disponible
if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor


# Création des fichiers train.pt et test.pt de MNIST
def create_MNIST():
    _ = dsets.MNIST(root='data/',
                    train=True, 
                    transform=transforms.ToTensor(),
                    download=True)
    shutil.move('data/processed/training.pt', 'data/train.pt')
    shutil.move('data/processed/test.pt', 'data/test.pt')
    shutil.rmtree('data/raw')
    shutil.rmtree('data/processed')


def train(nb_train=60000, flatten=False):
    if not os.path.exists('data/train.pt'):
        create_MNIST()
    images, labels = torch.load('data/train.pt')
    images, labels = images[:nb_train], labels[:nb_train]
    images = images.type(dtype) / 255
    if flatten:
        images = images.view(len(images), -1)
    else:
        images = images.view(len(images), 1, 28, 28)
    return images, labels


def test(nb_val=10000, flatten=False):
    if not os.path.exists('data/test.pt'):
        create_MNIST()
    images, labels = torch.load('data/test.pt')
    images, labels = images[:nb_val], labels[:nb_val]
    images = images.type(dtype) / 255
    if flatten:
        images = images.view(len(images), -1)
    else:
        images = images.view(len(images), 1, 28, 28)
    return images, labels
