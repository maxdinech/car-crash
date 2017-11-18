"""
Génération d'examples adversaires sur PyTorch

"""



import torch
from torch.autograd import Variable
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


def compare(image1, r, image2):
    fig = plt.figure()
    ax1 = fig.add_subplot(1,3,1)
    ax1.imshow(image1.data.view(28, 28).numpy(), cmap='gray')
    plt.title("Prédiction : {}".format(prediction(image1)))
    ax2 = fig.add_subplot(1,3,2)
    ax2.imshow(r.data.view(28, 28).numpy(), cmap='RdBu')
    plt.title("Perturbation")
    ax3 = fig.add_subplot(1,3,3)
    ax3.imshow(image2.data.view(28, 28).numpy(), cmap='gray')
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


def attaque_1(n, lr=0.005, div=-0.2):
    image = charge_image(n)
    chiffre = prediction(image)
    r = to_Var(torch.zeros(1, 1, 28, 28), requires_grad=True)
    image_adv = (image + r.clamp(-div, div)).clamp(0, 1)
    i = 0
    while prediction(image_adv) == chiffre:
        loss = model.forward(image_adv)[0,chiffre]
        loss.backward()
        print(str(i).zfill(3), loss.data[0], end='\r')
        r.data -= lr * r.grad.data / r.grad.data.abs().max()
        r.grad.data.zero_()
        image_adv = (image + r.clamp(-div, div)).clamp(0, 1)
        i += 1
        if i >= 300:
            break
    return (i < 300), image, r, image_adv


def attaque_2(n, lr=0.005, div=0.2):
    image = charge_image(n)
    chiffre = prediction(image)
    r = to_Var(torch.zeros(1, 1, 28, 28), requires_grad=True)
    image_adv = (image + (r * div / (1e-5 + r.norm()))).clamp(0, 1)
    i = 0
    while prediction(image_adv) == chiffre:
        loss = model.forward(image_adv)[0,chiffre]
        loss.backward()
        print(str(i).zfill(3), loss.data[0], end='\r')
        r.data -= lr * r.grad.data / r.grad.data.abs().max()
        r.grad.data.zero_()
        image_adv = (image + (r * div / (1e-5 + r.norm()))).clamp(0, 1)
        i += 1
        if i >= 300:
            break
    return (i < 300), image, r, image_adv


def attaque_optimale(n, a=0, b=5, lr=0.005):
    if b-a < 0.01:
        print("\n\nValeur minimale approchée : ", b)
        succes, image, r, image_adv = attaque_2(n, lr, b)
        compare(image, r, image_adv)
    else:
        c = (a+b)/2
        print("\n\n", c, "\n")
        succes, _, _, _ = attaque_2(n, lr, c)
        if succes:
            attaque_optimale(n, a, c, lr)
        else:
            attaque_optimale(n, c, b, lr)


def attaques(lr=0.005):
    for n in range(1000):
        attaque(n, lr)