"""
                        Tutoriel d'utilisation de Pytorch
                        =================================

Sommaire:
---------

    1. Manipulation des tenseurs ------------------------------------- 15
    2. Variables et gradients  --------------------------------------- 68
    3. Définition d'un MLP ------------------------------------------- 119

"""


#======================== 1. Manipulation des tenseurs ========================#

import torch

# On peut définir un tenseur depuis un tableau
x = torch.Tensor([2])
y = torch.Tensor([[1,2], [3,4]])


# Ou bien le définir uniquement par ses dimensions :
# matrice 5x3 non initialisée (valeurs VRAIMENT aléatoires)
x = torch.Tensor(5,3)
print(x)


# Ou encore : matrice 5x3 aléatoire (valeurs entre 0 et 1)
x = torch.rand(5,3)
print(x)


# Opérations sur les tenseurs
x = torch.rand(5,3)
y = torch.rand(5,3)
z = torch.rand(3,5)
print(x + y)  # addition. on aurait pu écrire torch.add(x, y)
print(x * y)  # produit de Hadamard (terme à terme)
print(x @ z)  # produit matriciel


# Fonctions définies sur les tenseurs
print(x.mean())
print(torch.sigmoid(x))
print(torch.log(x))
# et aussi : neg(), reciprocal(), pow(), sin(), tanh(), sqrt(), sign()...


# Les opérations "en place" modifient directement la valeur des tenseur
# Le caractère modificateur de ces opérations est symbolisé par le '_'.
x.add_(y)  # ajoute y à x
x.sub_(y)  # soustrait y de x


# Remise à zéro d'un tenseur (avec la même forme)
x.zero_()


# Redimensionnement des tenseurs (équivalent de la fonction reshape de numpy)
x = torch.range(1, 16)
print(x)
print(x.view(4, 4))



#========================= 2. Variables et gradients ==========================#

from torch.autograd import Variable

# Les variable ressemblent aux tenseurs, mais permettent en plus de calculer les
# gradients de manière automatisée
x = Variable(torch.rand(3,5))
print(x)


# Les variables se manipulent de la mème manière que les tenseurs
print(x + x)
print(x * x)


# On accède au tenseur contenu dans la variable avec le paramètre .data
print(x)
print(x.data)


# Quand un gradient a été calculé, on y accède avec .grad
print(x.grad)  # None : on n'a pas encore calculé de gradient pour x


# Variables de différentiation : paramètre 'requires_grad'
# (les dérivées partielles seront calculées par rapport à ces variables)
x = Variable(torch.Tensor([2]), requires_grad=True)


# Pour calculer un gradient il faut définir la fonction dont on calcule les
# gradients (jusqu'ici tout est logique).
y = x*x + 2
y.backward()   # gradients sur y -> création de x.grad
print(x.grad)  # On obtient bien 4 = 2*x


# Exemple de descente de gradient à une dimension : on souhaite trouver pour
# quelle valeur de x le min de y = x^2 - 2x est atteint.
x = Variable(torch.Tensor([2]), requires_grad=True)
for i in range(1000):
    y = x*x - 2*x                         # calcul de y = f(x)
    y.backward()                          # calcul des gradients
    x.data = x.data - 0.01 * x.grad.data  # descente du gradient
    x.grad.zero_()                        # effacement des anciens gradients
print("x =", x.data[0])

# Remarques : - 0n modifie la valeur des variables en modifiant variable.data.
#             - ATTENTION : Il est primordial d'effacer les anciens gradients !



#============================= 3. Un premier MLP ==============================#

# On va utiliser le module nn de pytorch pour se simplifier la vie :

from torch import nn


# Donées fictives d'entraînement : 5 valeurs.
# Les entrées sont dans R^3 et les sorties dans R^2
# Par convention les entrées s'appellent `x` et les sorties attendues `y`.
x = Variable(torch.randn(5, 3))
y = Variable(torch.randn(5, 2))


# Construction d'un réseau (modèle) sans couches cachées :
modele = nn.Linear(3, 2)
# Remarque : on aurait pu écrire directement modele = nn.Linear(3, 2) ici, mais
# la forme ci-dessus permet de généraliser 


# Acces direct aux poids et biais du modèle via .weight et .bias
print ('poids: ', modele.weight)
print ('biais: ', modele.bias)


# Calcul de la sortie du réseau (difficile de faire plus simple)
sortie = modele(x)


# Choix de la fonction d'erreur (criterion). Ici l'écart-type (MSE)
criterion = nn.MSELoss()


# Boucle de descente du gradient :
for i in range(100):
    # Calcul de l'erreur (loss).
    loss = criterion(modele(x), y)
    print('loss: ', loss.data[0])
    # Rétropropagation.
    loss.backward()
    # Ajustement des poids et des biais.
    modele.weight.data.sub_(0.01 * modele.weight.grad.data)
    modele.bias.data.sub_(0.01 * modele.bias.grad.data)
    # Remise à zéro des gradients
    modele.weight.grad.zero_()
    modele.bias.grad.zero_()
