"""Réseau MLP simple avec Torch pour la reconaissance de MNIST"""

import timeit
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F

# Hyperparamètres
# ---------------
lr = 1  # taux d'aprentissage
epochs = 1000
batch_size = 128


# Utilise automatiquement le GPU si CUDA est disponible
if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    ltype = torch.cuda.LongTensor
else:
    dtype = torch.FloatTensor
    ltype = torch.LongTensor


# Importation de la base de données et conversion npy -> tenseur
images = np.load('data/images.npy')
images.resize(60000, 28*28)
labels = np.load('data/labels.npy')
images = Variable(torch.from_numpy(images[:50000]).type(dtype), requires_grad=False)
labels = Variable(torch.from_numpy(labels[:50000]).type(dtype), requires_grad=False)


# On définit à la main un MLP avec deux couches cachées de 50 neurones
w1 = Variable(torch.randn(28*28, 50).type(dtype), requires_grad=True)
b1 = Variable(torch.randn(50).type(dtype), requires_grad=True)
w2 = Variable(torch.randn(50, 50).type(dtype), requires_grad=True)
b2 = Variable(torch.randn(50).type(dtype), requires_grad=True)
w3 = Variable(torch.randn(50, 10).type(dtype), requires_grad=True)
b3 = Variable(torch.randn(10).type(dtype), requires_grad=True)


start = timeit.default_timer()


for e in range(epochs):
    print("Epoch " + str(e+1) + " : ", end='')
    # Mélange de la BDD.
    ordre = torch.randperm(50000).type(ltype)
    images = images[ordre]
    labels = labels[ordre]

    for i in range(0, 50000, batch_size):
        x = images[i:i+batch_size]
        y = labels[i:i+batch_size]
        accs = []

        # Propagation de x dans le réseau. 
        a = F.relu(x @ w1 + b1)
        a = F.relu(a @ w2 + b2)
        a = F.softmax(a @ w3 + b3)
        y_pred = a

        # Calcul de l'erreur commise par le réseau : écart-type
        # loss = 1/(2n) * Somme(|y - y_pred|^2 pour y dans Y)
        loss = 5 * (y_pred - y).pow(2).mean()

        # Calcul et aff de la précision (pourcentage d'images bien classées)
        accs.append(100 * (torch.eye(10).type(dtype)[y_pred.data.max(1)[1]] * y.data).sum() / batch_size)

        # Rétropropagation du gradient sur l'erreur
        loss.backward()

        # Ajustement des poids selon la méthode de la descente de gradient.
        w1.data -= lr * w1.grad.data
        w2.data -= lr * w2.grad.data
        w3.data -= lr * w3.grad.data
        b1.data -= lr * b1.grad.data
        b2.data -= lr * b2.grad.data
        b3.data -= lr * b3.grad.data

        # Remise à zéro des gradients
        w1.grad.data.zero_()
        w2.grad.data.zero_()
        w3.grad.data.zero_()
        b1.grad.data.zero_()
        b2.grad.data.zero_()
        b3.grad.data.zero_()
    
    print(sum(accs)/len(accs), "%")

stop = timeit.default_timer()

print("\n\ntemps écoulé :", stop - start)

a = F.relu(images @ w1 + b1)
a = F.relu(a @ w2 + b2)
a = F.softmax(a @ w3 + b3)
y_pred = a

val_acc = 100 * (torch.eye(10)[y_pred.data.max(1)[1]] * labels.data).sum() / batch_size
print(val_acc)