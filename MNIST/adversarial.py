"""Génération d'examples adversaires sur Numpy"""


import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import mnist_loader


# Importation du modèle
try:
    model = torch.load('model.pt')
except FileNotFoundError:
    print("Pas de modèle existant !")


# Sélection d'une image aléatoire dans la base de données
