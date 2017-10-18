"""
Réseau adversaire avec PyTorch

Description : À partir d'une base de données R, on souhaite créer de fausses
données ressemblant à celles de R. Pour celà on fait finctionner en parallèle un
réseau discriminateur D et un réseau générateur G. Le réseau D s'entraîne pour
savoir si les données proviennent de G ou de R, et le réseau, en fonction de la
réponse de D, essaie de produire des faux plus convaincants.

Note : à aucun moment le réseau G n'a d'accès direct à la base R !

On a en plus besoin de définir I, une sorte de générateur de bruit aléatoire
utilisé par G.
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


epochs = 50
batch_size = 100
eta = 0.1

# On travaille sur des données aléatoires : un carré décentré en (3,0)

r_x = 2.5 + torch.rand(10_000, 1) / 2
r_y = 0.5 - torch.rand(10_000, 1) / 2
r = Variable(torch.cat((r_x, r_y), dim=1), requires_grad=False)


class Generator(nn.Module):
    def __init__(self, e, c, s):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(e, c)
        self.fc2 = nn.Linear(c, s)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class Discriminator(nn.Module):
    def __init__(self, e, c1, c2, s):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(e, c1)
        self.fc2 = nn.Linear(c1, c2)
        self.fc3 = nn.Linear(c2, s)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.sigmoid(self.fc3(x))


G = Generator(10, 10, 2)  # Autant de sorties que d'entrées dans D
D = Discriminator(2, 10, 10, 1)  # Une seule sortie : classification binaire

if torch.cuda.is_available():
    G = G.cuda()
    D = D.cuda()

loss_fn = F.mse_loss

def acc_fn(y_pred, y):
    return 100 * uns(len(y))[(y_pred - y).abs() < 0.5
    ].sum() / len(y)



def zeros(n):
    return Variable(torch.zeros(n, 1), requires_grad=False)

def uns(n):
    return Variable(torch.ones(n, 1), requires_grad=False)

def entropy(n):
    return Variable(torch.randn(n, 10), requires_grad=False)



for e in range(epochs):
    print("Epoch {}/{}:".format(e+1, epochs))
    perm = torch.randperm(len(r)).type(ltype)

    for i in range(0, len(r), batch_size):
        
        # 0. Affichage de la progression
        indice = str(min(i+batch_size, len(r))).zfill(5)
        print("└─ ({}/{}) ".format(indice, len(r)), end='')
        p = int(20 * i / (len(r) - batch_size))
        print('▰'*p + '▱'*(20-p), end='\r')
        
        # 1. Entraînement de D sur de vraies et fausses données
        D.zero_grad()
        
        real_data = r[perm[i:i+batch_size]]
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
    
    fake_data = G.forward(entropy(len(r)))
    
    D_pred_real = D.forward(fake_data)
    D_pred_fake = D.forward(fake_data)

    D_loss_real = loss_fn(D_pred_real, uns(len(r)))
    D_loss_fake = loss_fn(D_pred_fake, zeros(len(r)))
    D_loss = (D_loss_real + D_loss_fake).data[0] / 2

    D_acc_real = acc_fn(D_pred_real, uns(len(r)))
    D_acc_fake = acc_fn(D_pred_fake, zeros(len(r)))
    D_acc = (D_acc_real + D_acc_fake).data[0] / 2

    G_loss = loss_fn(D_pred_fake, uns(len(r))).data[0]
    G_acc = acc_fn(D_pred_fake, uns(len(r))).data[0]

    print("└─ ({0}/{0}) {1} ".format(len(r), '▰'*20), end='')
    print("D : loss: {:6.4f} - acc: {:5.2f}%  ─  ".format(D_loss, D_acc), end='')
    print("G : loss: {:6.4f} - acc: {:5.2f}%".format(G_loss, G_acc))

    r_x = r[:10000,0].data.numpy()
    r_y = r[:10000,1].data.numpy()
    plt.plot(r_x, r_y, 'r,')
    gen = G.forward(entropy(10000))
    x = gen[:,0].data.numpy()
    y = gen[:,1].data.numpy()
    plt.plot(x, y, 'b,')
    plt.plot([0, 0], [1.2, -1.2], 'w,')
    plt.show(block=False)
    time.sleep(1)
    plt.close()