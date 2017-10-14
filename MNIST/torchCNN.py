"""Réseau CNN simple avec Torch pour la reconaissance de MNIST"""

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


# Hyperparamètres
# ---------------
eta = 3  # taux d'aprentissage initial
epochs = 60
batch_size = 128
nb_train = 50_000
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
np_labels = np.load('data/labels.npy')
images = Variable(torch.from_numpy(np_images[:nb_train]).type(dtype), requires_grad=False)
labels = Variable(torch.from_numpy(np_labels[:nb_train]).type(dtype), requires_grad=False)
val_images = Variable(torch.from_numpy(np_images[-nb_val:]).type(dtype), requires_grad=False)
val_labels = Variable(torch.from_numpy(np_labels[-nb_val:]).type(dtype), requires_grad=False)
np_images, np_labels = 0, 0

images = images.view(len(images), 1, 28, 28)
val_images = val_images.view(len(val_images), 1, 28, 28)

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # convolutions : nb_canaux_entree, nb_canaux_sortie, dim_kernel
        self.conv1 = nn.Conv2d(1, 10, 3)
        self.conv2 = nn.Conv2d(10, 20, 3)
        self.fc1 = nn.Linear(20*5*5, 120)
        self.fc2 = nn.Linear(120, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(len(x), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if torch.cuda.is_available():
    model = Net().cuda()
else:
    model = Net()

loss_fn = F.mse_loss



print("Train on {} samples, validate on {} samples.".format(nb_train, nb_val))
print("Epochs: {}, batch_size: {}, eta: {}".format(epochs, batch_size, eta))
print()


for e in range(epochs):
    print("Epoch {}/{} - eta: {:5.3f}".format(e+1, epochs, eta))
    # Mélange de la BDD.
    ordre = torch.randperm(nb_train).type(ltype)

    for i in range(0, nb_train, batch_size):
        indice = str(min(i+batch_size, nb_train)).zfill(5)
        print("└─ ({}/{}) ".format(indice, nb_train), end='')
        p = int(20 * i / (nb_train - batch_size))
        print('▰'*p + '▱'*(20-p), end='\r')

        x = images[ordre[i:i+batch_size]]
        y = labels[ordre[i:i+batch_size]]

        # Propagation de x dans le réseau. 
        y_pred = model.forward(x)

        # Calcul de l'erreur commise par le réseau : écart-type
        loss = loss_fn(y_pred, y)

        # Remise à zéro des gradients avent la rétrop
        model.zero_grad()

        # Rétropropagation du gradient sur l'erreur
        loss.backward()

        # Ajustement des poids selon la méthode de la descente de gradient.
        for param in model.parameters():
            param.data -= eta * param.grad.data


    # Calcul et affichage de loss et acc
    
    y_pred = model.forward(images)
    acc = 100 * (y_pred.max(1)[1] == labels.max(1)[1]).data.sum() / nb_train
    loss = loss_fn(y_pred, labels).data[0]
    
    y_pred = model.forward(val_images)
    val_acc = 100 * (y_pred.max(1)[1] == val_labels.max(1)[1]).data.sum() / nb_val
    val_loss = loss_fn(y_pred, val_labels).data[0]

    print("└─ ({0}/{0}) {1} ".format(nb_train, '▰'*20), end='')
    print("loss: {:6.4f} - acc: {:5.2f}%  ─  ".format(loss, acc), end='')
    print("val_loss: {:6.4f} - val_acc: {:5.2f}%".format(val_loss, val_acc))


def ascii_print(image):
    for ligne in image:
        for pix in ligne:
            print(2*" ░▒▓█"[int(pix*5-0.001)%5], end='')
        print('')


def prediction(n):
    img = val_images[n].view(1, 28*28)
    pred = model.forward(img)
    print("prédiction :", pred.max(1)[1].data[0])
    ascii_print(img.view(28,28).data)


def prediction_img(img):
    pred = model.forward(img)
    print("prédiction :", pred.max(0)[1].data[0])
    ascii_print(img.view(28,28).data)