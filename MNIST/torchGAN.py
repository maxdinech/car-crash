"""
Réseau générateur adversaire avec PyTorch sur MNIST.
"""

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
import numpy as np

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    ltype = torch.cuda.LongTensor
else:
    dtype = torch.FloatTensor
    ltype = torch.LongTensor


# Hyperparamètres
epochs = 200
batch_size = 10
G_lr = 0.0003
D_lr = 0.0003
chiffre = 3

# Chargement de la BDD
images = np.load('data/images.npy')
labels = np.load('data/labels.npy')
images = np.array([images[i] for i in range(len(images)) if labels[i,chiffre] == 1])
images = torch.from_numpy(images).type(dtype)
labels = 0

data_loader = torch.utils.data.DataLoader(images,
                                          batch_size=10,
                                          shuffle=True)

# Conversion simple tensor -> Variable
def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


# Discriminateur
D = nn.Sequential(
        nn.Linear(784, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 1),
        nn.Sigmoid())


# Générateur 
G = nn.Sequential(
        nn.Linear(64, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 784),
        nn.Sigmoid())


# # Discriminateur
# D = nn.Sequential(
#         nn.Linear(784, 256),
#         nn.LeakyReLU(0.2),
#         nn.Linear(256, 256),
#         nn.LeakyReLU(0.2),
#         nn.Linear(256, 1),
#         nn.Sigmoid())


# # Générateur 
# G = nn.Sequential(
#         nn.Linear(64, 256),
#         nn.LeakyReLU(0.2),
#         nn.Linear(256, 256),
#         nn.LeakyReLU(0.2),
#         nn.Linear(256, 784),
#         nn.Sigmoid())


# Déplacement vers le GPU si possible
if torch.cuda.is_available():
    D.cuda()
    G.cuda()


# loss : Binary cross entropy loss
# BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
loss_fn = nn.BCELoss()


# Optimiseur : Adam
D_optimizer = torch.optim.Adam(D.parameters(), lr=D_lr)
G_optimizer = torch.optim.Adam(G.parameters(), lr=G_lr)


# Générateur de bruit aléatoire z
def entropy(n):
    return Variable(torch.randn(n, 64).type(dtype), requires_grad=False)


# Affichage d'image
def ascii_print(image):
    image = image.view(28,28)
    for ligne in image:
        for pix in ligne:
            print(2*" ░▒▓█"[int(pix*4.999)%5], end='')
        print('')


# Affichages multiples
def ascii_print_sided(img1, img2):
    img1 = img1.view(28,28)
    img2 = img2.view(28,28)
    image = torch.cat((img1, img2), dim = 1)
    for ligne in image:
        for pix in ligne:
            print(2*" ░▒▓█"[int(pix*4.999)%5], end='')
        print('')


# Entraînement de D et G
for epoch in range(epochs):
    print("Epoch {}/{}:".format(epoch + 1, epochs))

    for i, images in enumerate(data_loader):
        
        indice = str((i+1)).zfill(3)
        print("└─ ({}/{}) ".format(indice, len(data_loader)), end='')
        p = int(20 * i / len(data_loader))
        print('▰'*p + '▱'*(20-p), end='   ')
        
        # taille locale du mini-batch
        local_batch_size = len(images)

        # Labels des vraies et fausses entrées
        real_labels = to_var(torch.ones(local_batch_size, 1))
        fake_labels = to_var(torch.zeros(local_batch_size, 1))

        #=============== entraînement de D ===============#
        real_images = to_var(images.view(local_batch_size, -1))
        D_pred_real = D(real_images)
        D_loss_real = loss_fn(D_pred_real, real_labels)

        fake_images = G(entropy(local_batch_size))
        D_pred_fake = D(fake_images)
        D_loss_fake = loss_fn(D_pred_fake, fake_labels)
        
        D_loss = D_loss_real + D_loss_fake

        D.zero_grad()
        D_loss.backward()
        D_optimizer.step()

        #=============== entraînement de G ===============#

        fake_images = G(entropy(local_batch_size))
        D_pred_fake = D(fake_images)
        G_loss = loss_fn(D_pred_fake, real_labels)

        G.zero_grad()
        G_loss.backward()
        G_optimizer.step()


        print('D_loss: %.4f,   G_loss: %.4f,   D(x): %.2f,   D(G(z)): %.2f' 
              %(D_loss.data[0], G_loss.data[0],
                D_pred_real.data.mean(), D_pred_fake.data.mean()), end='\r')

    # Calcul et affichage de loss et acc à chaque fin d'epoch
    
    img1 = G.forward(entropy(1)).data
    img2 = G.forward(entropy(1)).data
    ascii_print_sided(img1, img2)


while True:
    print("\033[H\033[J")
    img1 = G.forward(entropy(1)).data
    img2 = G.forward(entropy(1)).data
    ascii_print_sided(img1, img2)
    time.sleep(0.7)