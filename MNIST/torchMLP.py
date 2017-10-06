"""Réseau MLP simple avec Torch"""


import torch
from torch.autograd import Variable


# Utilise automatiquement le GPU si CUDA est disponible
if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor


E, C, S = 1000, 100, 10  # Dimension des couches entrée, cachée et sortie
taille_données = 1000


# Tenseurs random, dans des Variables, qui représentent la BDD d'entraînement.
# On spécifie `requires_grad=False` pour dire que les calculs de gradients ne
# considèreront pas des valeurs commes variables de différentiation
x = Variable(torch.rand(taille_données, E).type(dtype), requires_grad=False)
y = Variable(torch.rand(taille_données, S).type(dtype), requires_grad=False)


# Tenseurs random normalisées, dans des Variables, représentant poids et biais.
# Cette fois-ci on spécifie `requires_grad=True` : on aimerait calculer les
# gradients des poids et des biais.
w1 = Variable(torch.randn(E, C).type(dtype), requires_grad=True)
b1 = Variable(torch.randn(C).type(dtype), requires_grad=True)
w2 = Variable(torch.randn(C, S).type(dtype), requires_grad=True)
b2 = Variable(torch.randn(S).type(dtype), requires_grad=True)


learning_rate = 1e-3

for t in range(500):
    # Propagation de x dans le réseau ; le `.clamp(min=0)` représente ReLU. 
    y_pred = ((x @ w1 - b1).clamp(min=0)) @ w2 - b2

    # Calcul de l'erreur commise par le réseau : écart-type
    loss = (y_pred - y).pow(2).mean()
    print(t, loss.data[0])

    # autograd va maintenant calculer les gradients de toutes les variables
    # intervenant dans l'expression de loss et pour lesquelles on a spécifié
    # `requires_grad=True`. -> création de w1.grad, b1.grad, ... qui contiennent
    # les dérivées pertielles de loss.
    loss.backward()

    # Ajustement des poids selon la méthode de la descente de gradient.
    w1.data -= learning_rate * w1.grad.data
    w2.data -= learning_rate * w2.grad.data
    b1.data -= learning_rate * b1.grad.data
    b2.data -= learning_rate * b2.grad.data

    # Remise à zéro des gradients avant tout recalcul !
    w1.grad.data.zero_()
    w2.grad.data.zero_()
    b1.grad.data.zero_()
    b2.grad.data.zero_()
