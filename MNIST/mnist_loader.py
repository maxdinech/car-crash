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
    # On appelle `val` les images de `test`
    shutil.move('data/processed/test.pt', 'data/val.pt')
    # On divise `training` en `train` et `test`
    images, labels = torch.load('data/processed/training.pt')
    train_images, train_labels = images[:50000], labels[:50000]
    val_images, val_labels = images[50000:], labels[50000:]
    torch.save((train_images, train_labels), "data/train.pt")
    torch.save((val_images, val_labels), "data/test.pt")
    # On supprimme les dossiers temporaires
    shutil.rmtree('data/raw')
    shutil.rmtree('data/processed')


def train(nb_train=50000):
    if not os.path.exists('data/train.pt'):
        create_MNIST()
    images, labels = torch.load('data/train.pt')
    images, labels = images[:nb_train], labels[:nb_train]
    images = images.type(dtype) / 255
    images = images.view(len(images), 1, 28, 28)
    return images, labels


def test(nb_test=10000):
    if not os.path.exists('data/test.pt'):
        create_MNIST()
    images, labels = torch.load('data/test.pt')
    images, labels = images[:nb_test], labels[:nb_test]
    images = images.type(dtype) / 255
    images = images.view(len(images), 1, 28, 28)
    return images, labels


def val(nb_val=10000):
    if not os.path.exists('data/val.pt'):
        create_MNIST()
    images, labels = torch.load('data/val.pt')
    images, labels = images[:nb_val], labels[:nb_val]
    images = images.type(dtype) / 255
    images = images.view(len(images), 1, 28, 28)
    return images, labels
