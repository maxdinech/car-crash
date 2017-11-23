"""
AlphaGo
=======

Trois choses : deux réseaux et une méthode d'exploration d'arbre.

- Le premier (Policy Network) indique les cases "intéressantes", sur lesquelles
  on va faire une exploration d'arbre.

- Le deuxième (Value Network) estime les chances de gagner des deux joueurs. On
  l'utilise dans l'exploration de l'arbre pour déterminer la situation la plus
  avantageuse.

- L'exploration d'arbre se fait en Monte-Carlo pour gagner en vitesse.


AlphaGo Zero
============

Fusion des deux réseaux en un seul, et pas d'utilisation de "vraie" partie pour
s'entraîner.

"""

import torch
from torch.autograd import Variable
from torch import nn

# Morpion sur une grille 10x10
# Il faut aligner quatre jetons pour gagner

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        # convolutions : nb_canaux_entree, nb_canaux_sortie, dim_kernel
        self.conv = nn.Conv2d(1, 20, 4)
        self.fc1 = nn.Linear(7*7*20, 128)
        self.fc2 = nn.Linear(128, 100)
        self.fc3 = nn.Linear(100, 43)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(len(x), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x))
        return x

