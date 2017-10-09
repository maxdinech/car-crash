"""Réseau MLP simple avec Torch pour la reconaissance de MNIST"""

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F

# Hyperparamètres
# ---------------
eta = 0.5  # taux d'aprentissage
epochs = 400
batch_size = 10
nb_train = 10_000
nb_val = 10_000

# Utilise automatiquement le GPU si CUDA est disponible
if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    ltype = torch.cuda.LongTensor
else:
    dtype = torch.FloatTensor
    ltype = torch.LongTensor


# Importation de la base de données et conversion npy -> tenseur
np_images = np.load('data/images.npy')
np_images.resize(60000, 28*28)
np_labels = np.load('data/labels.npy')
images = Variable(torch.from_numpy(np_images[:nb_train]).type(dtype), requires_grad=False)
labels = Variable(torch.from_numpy(np_labels[:nb_train]).type(dtype), requires_grad=False)
val_images = Variable(torch.from_numpy(np_images[-nb_val:]).type(dtype), requires_grad=False)
val_labels = Variable(torch.from_numpy(np_labels[-nb_val:]).type(dtype), requires_grad=False)



# On définit à la main un MLP avec une couche cachée de 30 neurones
w1 = Variable(torch.randn(28*28, 30).type(dtype), requires_grad=True)
b1 = Variable(torch.randn(30).type(dtype), requires_grad=True)
w2 = Variable(torch.randn(30, 10).type(dtype), requires_grad=True)
b2 = Variable(torch.randn(10).type(dtype), requires_grad=True)

print("Train on {} samples, validate on {} samples.".format(nb_train, nb_val))
print("Epochs : {}, batch_size : {}, eta: {}".format(epochs, batch_size, eta))
print()


for e in range(epochs):
    # Mélange de la BDD.
    ordre = torch.randperm(nb_train).type(ltype)
    images = images[ordre]
    labels = labels[ordre]

    for i in range(0, nb_train, batch_size):
        print("Epoch " + str(e+1) + " : (", str(i+batch_size).zfill(5), "/{})".format(nb_train), sep='', end='\r')

        x = images[i:i+batch_size]
        y = labels[i:i+batch_size]

        # Propagation de x dans le réseau. 
        a = F.relu(x @ w1 + b1)
        a = F.softmax(a @ w2 + b2)
        y_pred = a

        # Calcul de l'erreur commise par le réseau : écart-type
        # -> équivaut à F.mse_loss
        # loss = (y_pred - y).pow(2).mean()
        loss = F.cross_entropy(y_pred, y.max(1)[1])

        # Rétropropagation du gradient sur l'erreur
        loss.backward()

        # Ajustement des poids selon la méthode de la descente de gradient.
        w1.data -= eta * w1.grad.data
        w2.data -= eta * w2.grad.data
        b1.data -= eta * b1.grad.data
        b2.data -= eta * b2.grad.data

        # Remise à zéro des gradients
        w1.grad.data.zero_()
        w2.grad.data.zero_()
        b1.grad.data.zero_()
        b2.grad.data.zero_()
    
    y_pred = F.softmax(F.relu(images @ w1 + b1) @ w2 + b2)
    train_acc = 100 * (y_pred.max(1)[1] == labels.max(1)[1]).data.sum() / nb_train
    y_pred = F.softmax(F.relu(val_images @ w1 + b1) @ w2 + b2)
    val_acc = 100 * (y_pred.max(1)[1] == val_labels.max(1)[1]).data.sum() / nb_val

    print("Epoch " + str(e+1) + " : ({0}/{0}) -- train_acc:".format(nb_train), train_acc, "%,  val_acc:", val_acc, "%", sep='')


def ascii_print(image):
    for ligne in image.data:
        for pix in ligne:
            print(2*" ░▒▓█"[int(pix*5-0.001)], end='')
        print('')


def prediction(n):
    img = val_images[n].view(1, 28*28)
    pred = F.softmax(F.relu(img @ w1 + b1) @ w2 + b2)
    print("prédiction :", pred.max(1)[1].data[0])
    ascii_print(img.view(28,28))