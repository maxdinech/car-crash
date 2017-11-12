"""
CNN avec PyTorch sur MNIST.

Résultats attendus : 99.5 %

TODO:
    - Mettre en place un dropout après chaque Dense.

"""


import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import gtsrb_loader


# Hyperparamètres
# ---------------
couleur = 'clahe'
eta = 0.0003
epochs = 30
batch_size = 128
nb_train = 6000
nb_val = 1000


# Création de variable sur GPU si possible, CPU sinon
def to_Var(x, requires_grad=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


# Chargement des bases de données
train_images, train_labels = gtsrb_loader.train(couleur, nb_train)
test_images, test_labels = gtsrb_loader.test(couleur, nb_val)

# Création du DataLoader
train_loader = DataLoader(TensorDataset(train_images, train_labels),
                          batch_size=batch_size,
                          shuffle=True)

nb_batches = len(train_loader)

# Conversion des BDD en Variables
train_images = to_Var(train_images)
train_labels = to_Var(train_labels)
test_images = to_Var(test_images)
test_labels = to_Var(test_labels)


# Définition du modèle : CNN à deux convolutions
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # convolutions : nb_canaux_entree, nb_canaux_sortie, dim_kernel
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(20, 40, 3)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(8*8*40, 120)
        self.fc2 = nn.Linear(120, 10)

    def forward(self, x):
        x = x.view(len(x), 1, 40, 40)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(len(x), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x))
        return x


# Génération d'un CNN initialisé aléatoirement
model = CNN()

# Déplacement vers le GPU si possible
if torch.cuda.is_available():
    model = model.cuda()


# Fonction d'erreur
loss_fn = nn.CrossEntropyLoss()


# Optimiseur (méthode utilisée pour diminuer l'erreur)
optimizer = torch.optim.Adam(model.parameters(), lr=eta)


# Fonction de calcul de la précision du réseau
def accuracy(y_pred, y):
    return 100 * (y_pred.max(1)[1] == y).data.sum() / len(y)


print("Train on {} samples, validate on {} samples.".format(nb_train, nb_val))
print("Epochs: {}, batch_size: {}, eta: {}\n".format(epochs, batch_size, eta))


# Boucle principale sur chaque epoch
for e in range(epochs):

    print("Epoch {}/{}".format(e+1, epochs))

    # Boucle secondaire sur chaque mini-batch
    for i, (x, y) in enumerate(train_loader):

        batch = str((i+1)).zfill(len(str(nb_batches)))
        print("└─ ({}/{}) ".format(batch, nb_batches), end='')
        p = int(20 * i / nb_batches)
        print('▰'*p + '▱'*(20-p), end='\r')
        
        # Propagation dans le réseau et calcul de l'erreur
        y_pred = model.forward(to_Var(x))
        loss = loss_fn(y_pred, to_Var(y))

        # Ajustement des paramètres
        model.zero_grad()
        loss.backward()
        optimizer.step()

    # Calcul de l'erreur totale et de la précision sur la base d'entraînement
    y_pred = model.forward(train_images)
    acc = accuracy(y_pred, train_labels)
    loss = loss_fn(y_pred, train_labels).data[0]
    
    # Calcul de l'erreur totale et de la précision sur la base de validation
    y_pred = model.forward(test_images)
    val_acc = accuracy(y_pred, test_labels)
    val_loss = loss_fn(y_pred, test_labels).data[0]

    print("└─ ({0}/{0}) {1} ".format(nb_batches, '▰'*20), end='')
    print("loss: {:6.4f} - acc: {:5.2f}%  ─  ".format(loss, acc), end='')
    print("val_loss: {:6.4f} - val_acc: {:5.2f}%".format(val_loss, val_acc))


# Enregistrement du réseau
torch.save(model, 'model.pt')


def ascii_print(image):
    image = image.view(40,40)
    for ligne in image:
        for pix in ligne:
            print(2*" ░▒▓█"[int(pix*4.999)%5], end='')
        print('')


def prediction(n):
    img = test_images[n].view(1, 1, 40, 40)
    pred = model.forward(img)
    print("prédiction :", pred.max(1)[1].data[0])
    ascii_print(img.data)


def prediction_img(img):
    pred = model.forward(img)
    print("prédiction :", pred.max(0)[1].data[0])
    ascii_print(img.data)


import random, time
def affichages():
    while True:
        print("\033[H\033[J")
        prediction(random.randrange(1000))
        time.sleep(0.7)
