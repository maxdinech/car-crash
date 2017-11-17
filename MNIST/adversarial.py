"""
Génération d'examples adversaires sur PyTorch

"""



import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import mnist_loader
import matplotlib.pyplot as plt



# Création de variable sur GPU si possible, CPU sinon
def to_Var(x, requires_grad=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


# Importation du modèle
try:
    if torch.cuda.is_available():
        model = torch.load('model.pt').cuda()
    else:
        model = torch.load('model.pt', map_location=lambda storage, loc: storage)
except FileNotFoundError:
    print("Pas de modèle trouvé !")


def compare(image1, image2):
    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    ax1.imshow(image1.data.view(28, 28).numpy(), cmap='gray')
    plt.title("Prédiction : {}".format(prediction(image1)))
    ax2 = fig.add_subplot(1,2,2)
    ax2.imshow(image2.data.view(28, 28).numpy(), cmap='gray')
    plt.title("Prédiction : {}".format(prediction(image2)))
    plt.show()

def affiche(image):
    plt.imshow(image.clamp(0, 1).data.view(28, 28).numpy(), cmap='gray')
    plt.show()

# Sélection d'une image dans la base de données (la première : un chiffre 5)


# Première méthode d'attaque : exploration du voisinage de l'image en cherchant
# à avoir une grande erreur tout en restant assez près On cherche la direction
# la plus favorable par calcul du gradient
#   1. On calcule et trie les gradients
#   2. On modifie l'image
# Jusqu'à obtenir une prédiction incorrecte


def charge_image(n):
    images, _ = mnist_loader.train(n+1)
    return to_Var(images[n].view(1, 1, 28, 28))


def prediction(image):
    return model.forward(image.clamp(0, 1)).max(1)[1].data[0]


def attaque(n, eta=0.005):
    image = charge_image(n)
    chiffre = prediction(image)
    r = to_Var(torch.zeros(1, 1, 28, 28), requires_grad=True)
    image_adv = (image + r).clamp(0, 1)
    i = 0
    while prediction(image_adv) == chiffre:
        loss = model.forward(image_adv)[0,chiffre]
        loss.backward()
        print(loss.data[0])
        r.data -= eta * r.grad.data / r.grad.data.abs().max()
        r.grad.data.zero_()
        image_adv = (image + r).clamp(0, 1)
        i += 1
        if i > 1000:
            break
    compare(image, image_adv)


def attaque_2(n, eta=0.005):
    image = charge_image(n)
    chiffre = prediction(image)
    r = to_Var(torch.zeros(1, 1, 28, 28), requires_grad=True)
    demis = to_Var(torch.ones(1, 1, 28, 28) / 2)
    image_adv = (1 - r.clamp(0, 1)) * image + r.clamp(0, 1) * demis
    i = 0
    while prediction(image_adv) == chiffre:
        loss = model.forward(image_adv)[0,chiffre]
        loss.backward()
        print(loss.data[0])
        r.data -= eta * r.grad.data / r.grad.data.abs().max()
        r.grad.data.zero_()
        image_adv = (1 - r.clamp(0, 1)) * image + r.clamp(0, 1) * demis
        i += 1
        if i > 1000:
            break
    compare(image, image_adv)


def attaques(eta=0.005):
    for n in range(1000):
        attaque(n, eta)