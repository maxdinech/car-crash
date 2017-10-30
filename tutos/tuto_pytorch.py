"""
                        Tutoriel d'utilisation de Pytorch
                        =================================

Sommaire:
---------

    1. Manipulation des tenseurs ------------------------------------- 16
    2. Variables et gradients  --------------------------------------- 69
    3. Définition d'un MLP simple ------------------------------------ 120
    4. Définition d'un MLP avec le module torch.nn ------------------- 176

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



#======================= 3. Définition d'un MLP simple  =======================#

# On applique les mêmes principes que précedemment pour optimiser un réseau MLP
# à 1000 entrées, une couche cachée de 100 neurones et 10 sorties.

E, C, S = 30, 100, 10  # Dimension des couches entrée, cachée et sortie


# Tenseurs random, dans des Variables, qui représentent la BDD d'entraînement.
# On spécifie `requires_grad=False` pour dire que les calculs de gradients ne
# considèreront pas des valeurs commes variables de différentiation
# Le 64 représente la taille de la base de données.
x = Variable(torch.rand(64, E), requires_grad=False)
y = Variable(torch.rand(64, S), requires_grad=False)


# Tenseurs random normalisées, dans des Variables, représentant poids et biais.
# Cette fois-ci on spécifie `requires_grad=True` : on aimerait calculer les
# gradients des poids et des biais.
w1 = Variable(torch.randn(E, C), requires_grad=True)
b1 = Variable(torch.randn(C), requires_grad=True)
w2 = Variable(torch.randn(C, S), requires_grad=True)
b2 = Variable(torch.randn(S), requires_grad=True)


# Boucle d'aprentissage
for t in range(500):
    # Propagation de x dans le réseau.
    out = (x @ w1) + b1
    out = out.clamp(min=0)  # ReLU sur `out`
    out = (out @ w2) + b2
    y_pred = out

    # Calcul de l'erreur commise par le réseau : écart-type
    loss = (y_pred - y).pow(2).mean()
    print(t, loss.data[0])

    # autograd va maintenant calculer les gradients de toutes les variables
    # intervenant dans l'expression de loss et pour lesquelles on a spécifié
    # `requires_grad=True` : création de w1.grad, b1.grad, w2.grad et b2.grad
    # qui contiennent les dérivées partielles de loss.
    loss.backward()

    # Ajustement des poids selon la méthode de la descente de gradient.
    w1.data -= 1e-3 * w1.grad.data
    w2.data -= 1e-3 * w2.grad.data
    b1.data -= 1e-3 * b1.grad.data
    b2.data -= 1e-3 * b2.grad.data

    # Remise à zéro des gradients avant tout recalcul !
    w1.grad.data.zero_()
    w2.grad.data.zero_()
    b1.grad.data.zero_()
    b2.grad.data.zero_()



#======================= 4. MLP avec le module torch.nn =======================#

# On se simplifie la vie en utilisant le module torch.nn, et on définit la
# classe Net du réseau voulu comme sous-classe de nn.Module. Par exemple :

class Net(nn.Module):

    def __init__(self, e, c, s):
        super(Net, self).__init__()
        # convolutions : nb_canaux_entree, nb_canaux_sortie, dim_kernel
        self.couche1 = nn.Linear(e, c)
        self.couche2 = nn.Linear(c, s)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x

reseau_de_neurones = Net(30, 100, 10)

# Tout se fait exactement comme précedemment. les sorties calculées pour une
# entrée x s'obtiennent par reseau_de_neurones.forward(x)
# Pour l'ajustement des paramètres, on utilise :

for param in reseau_de_neurones.parameters():
    param.data -= 1e-3 * param.grad.data
