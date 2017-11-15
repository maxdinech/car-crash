"""Génération d'examples adversaires sur PyTorch"""


import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import mnist_loader


# Définition du modèle : CNN à deux convolutions
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # convolutions : nb_canaux_entree, nb_canaux_sortie, dim_kernel
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(20, 40, 3)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(5*5*40, 120)
        self.fc2 = nn.Linear(120, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(len(x), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x))
        return x


# Importation du modèle
try:
    if torch.cuda.is_available():
        model = torch.load('model.pt').cuda()
    else:
        torch.load('model.pt', map_location=lambda storage, loc: storage)
except FileNotFoundError:
    print("Pas de modèle existant !")


# Sélection d'une image aléatoire dans la base de données

image, label = mnist_loader.train(1)