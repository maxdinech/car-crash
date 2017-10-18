"""
Réseau générateur adversaire avec PyTorch sur MNIST.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import time

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    ltype = torch.cuda.LongTensor
else:
    dtype = torch.FloatTensor
    ltype = torch.LongTensor


epochs = 200
batch_size = 10
eta = 0.01


images = np.load('data/images.npy')
labels = np.load('data/labels.npy')
images = np.array([images[i] for i in range(len(images)) if labels[i,3] == 1])
images = torch.from_numpy(images).type(dtype)
images = Variable(images, requires_grad=False)


class Generator(nn.Module):
    def __init__(self, e, c, s):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(e, c)
        self.fc2 = nn.Linear(c, s)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x.view(len(x), 28, 28)


class Discriminator(nn.Module):
    def __init__(self, e, c1, c2, s):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(e, c1)
        self.fc2 = nn.Linear(c1, c2)
        self.fc3 = nn.Linear(c2, s)
    def forward(self, x):
        x = x.view(len(x), 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.sigmoid(self.fc3(x))


G = Generator(100, 100, 28*28)  # Autant de sorties que d'entrées dans D
D = Discriminator(28*28, 100, 30, 1)  # Une seule sortie : classification binaire

if torch.cuda.is_available():
    G = G.cuda()
    D = D.cuda()

loss_fn = F.binary_cross_entropy


def acc_fn(y_pred, y):
    return 100 * uns(len(y))[(y_pred - y).abs() < 0.5].sum() / len(y)



def zeros(n):
    return Variable(torch.zeros(n, 1).type(dtype), requires_grad=False)

def uns(n):
    return Variable(torch.ones(n, 1).type(dtype), requires_grad=False)

def entropy(n):
    return Variable(torch.randn(n, 100).type(dtype), requires_grad=False)


def ascii_print(image):
    image = image.view(28,28)
    for ligne in image:
        for pix in ligne:
            print(2*" ░▒▓█"[int(pix*4.999)%5], end='')
        print('')


def ascii_print_sided(img1, img2, img3):
    img1 = img1.view(28,28)
    img2 = img2.view(28,28)
    img3 = img3.view(28,28)
    image = torch.cat((img1, img2, img3), dim = 1)
    for ligne in image:
        for pix in ligne:
            print(2*" ░▒▓█"[int(pix*4.999)%5], end='')
        print('')


def prediction(n):
    img = val_images[n].view(1, 1, 28, 28)
    pred = model.forward(img)
    print("prédiction :", pred.max(1)[1].data[0])
    ascii_print(img.data)


def prediction_img(img):
    pred = model.forward(img)
    print("prédiction :", pred.max(0)[1].data[0])
    ascii_print(img.data)


for e in range(epochs):
    print("Epoch {}/{}:".format(e+1, epochs))
    perm = torch.randperm(len(images)).type(ltype)

    for i in range(0, len(images) - batch_size + 1, batch_size):
        
        # 0. Affichage de la progression
        indice = str(min(i+batch_size, len(images))).zfill(5)
        print("└─ ({}/{}) ".format(indice, len(images)), end='')
        p = int(20 * i / (len(images) - batch_size))
        print('▰'*p + '▱'*(20-p), end='\r')
        
        # 1. Entraînement de D sur de vraies et fausses données
        D.zero_grad()
        
        real_data = images[perm[i:i+batch_size]]
        D_pred = D.forward(real_data)
        loss = loss_fn(D_pred, uns(batch_size))
        loss.backward()

        fake_data = G.forward(entropy(batch_size))
        D_pred = D.forward(fake_data)
        loss = loss_fn(D_pred, zeros(batch_size))
        loss.backward()

        for param in D.parameters():
            param.data -= eta * param.grad.data


        # 2. Entraînement de G à partir du nouveau D
        G.zero_grad()

        fake_data = G.forward(entropy(2 * batch_size))
        D_pred = D.forward(fake_data)
        loss = loss_fn(D_pred, uns(2 * batch_size))
        loss.backward()

        for param in G.parameters():
            param.data -= eta * param.grad.data


    # Calcul et affichage de loss et acc à chaque fin d'epoch
    
    fake_data = G.forward(entropy(len(images)))
    
    D_pred_real = D.forward(fake_data)
    D_pred_fake = D.forward(fake_data)

    D_loss_real = loss_fn(D_pred_real, uns(len(images)))
    D_loss_fake = loss_fn(D_pred_fake, zeros(len(images)))
    D_loss = (D_loss_real + D_loss_fake).data[0] / 2

    D_acc_real = acc_fn(D_pred_real, uns(len(images)))
    D_acc_fake = acc_fn(D_pred_fake, zeros(len(images)))
    D_acc = (D_acc_real + D_acc_fake).data[0] / 2

    G_loss = loss_fn(D_pred_fake, uns(len(images))).data[0]
    G_acc = acc_fn(D_pred_fake, uns(len(images))).data[0]

    print("└─ ({0}/{0}) {1} ".format(len(images), '▰'*20), end='')
    print("D : loss: {:6.4f} - acc: {:5.2f}%  ─  ".format(D_loss, D_acc), end='')
    print("G : loss: {:6.4f} - acc: {:5.2f}%".format(G_loss, G_acc))

    img1 = G.forward(entropy(1)).data
    img2 = G.forward(entropy(1)).data
    img3 = G.forward(entropy(1)).data
    ascii_print_sided(img1, img2, img3)


while True:
    print("\033[H\033[J")
    img1 = G.forward(entropy(1)).data
    img2 = G.forward(entropy(1)).data
    img3 = G.forward(entropy(1)).data
    ascii_print_sided(img1, img2, img3)
    time.sleep(0.7)