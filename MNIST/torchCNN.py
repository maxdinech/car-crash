"""Réseau CNN simple avec Torch pour la reconaissance de MNIST"""


import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import mnist_loader


# Hyperparamètres
# ---------------
eta = 3  # taux d'aprentissage initial
epochs = 30
batch_size = 128
nb_train = 60000
nb_val = 10000


# Utilise automatiquement le GPU si CUDA est disponible
def to_Var(x, requires_grad=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


# Chargement des bases de données
train_images, train_labels = mnist_loader.train(nb_train)
test_images, test_labels = mnist_loader.test(nb_val)


# Création du DataLodaer de train
train_loader = DataLoader(TensorDataset(train_images, train_labels),
                          batch_size=batch_size,
                          shuffle=True)


# Définition du réseau : CNN à deux convolutions
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # convolutions : nb_canaux_entree, nb_canaux_sortie, dim_kernel
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(20, 40, 3)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(5*5*40, 120)
        self.fc2 = nn.Linear(120, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(len(x), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x))
        return x

model = Net()

if torch.cuda.is_available():
    model = model.cuda()

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=eta)

def accuracy(y_pred, y):
    return 100 * (y_pred.max(1)[1] == y.max(1)[1]).data.sum() / len(y)


print("Train on {} samples, validate on {} samples.".format(nb_train, nb_val))
print("Epochs: {}, batch_size: {}, eta: {}".format(epochs, batch_size, eta))
print()


for e in range(epochs):

    print("Epoch {}/{} - eta: {:5.3f}".format(e+1, epochs, eta))

    for i, (x, y) in enumerate(train_loader):

        indice = str((i+1)).zfill(3)
        print("└─ ({}/{}) ".format(indice, len(train_loader)), end='')
        p = int(20 * i / len(train_loader))
        print('▰'*p + '▱'*(20-p), end='\r')
        
        y_pred = model.forward(to_Var(x))
        loss = loss_fn(y_pred, to_Var(y))

        model.zero_grad()
        loss.backward()
        optimizer.step()

    # Calcul et affichage de loss et acc à chaque fin d'epoch
    
    y_pred = model.forward(train_images)
    acc = accuracy(y_pred, train_labels)
    loss = loss_fn(y_pred, train_labels).data[0]
    
    y_pred = model.forward(test_images)
    val_acc = accuracy(y_pred, test_labels)
    val_loss = loss_fn(y_pred, test_labels).data[0]

    print("└─ ({0}/{0}) {1} ".format(nb_train, '▰'*20), end='')
    print("loss: {:6.4f} - acc: {:5.2f}%  ─  ".format(loss, acc), end='')
    print("val_loss: {:6.4f} - val_acc: {:5.2f}%".format(val_loss, val_acc))


def ascii_print(image):
    image = image.view(28,28)
    for ligne in image:
        for pix in ligne:
            print(2*" ░▒▓█"[int(pix*4.999)%5], end='')
        print('')


def prediction(n):
    img = test_images[n].view(1, 1, 28, 28)
    pred = model.forward(img)
    print("prédiction :", pred.max(1)[1].data[0])
    ascii_print(img.data)


def prediction_img(img):
    pred = model.forward(img)
    print("prédiction :", pred.max(0)[1].data[0])
    ascii_print(img.data)


def affichages():
    while True:
        print("\033[H\033[J")
        prediction(random.randrange(1000))
        time.sleep(0.7)


# ------------------------------

# image = Variable(val_images[0].data.clone(), requires_grad=True)
# for param in model.parameters():
#     param.data -= eta * param.grad.data

# def adversaire(image, n):
#     image = Variable(val_images[num_image].data.clone(), requires_grad=True)
#     while prediction_bis(image) != n:
#         loss_bis = loss_fn_bis(propagation_bis(image), n)
#         loss_bis.backward()
#         pos_max = image.grad.data.max(0)[1][0]
#         image[pos_max].data -= 10 * image.grad.data[pos_max]
#         image.grad.data = torch.zeros(28*28)
#         print(pos_max)
#     prediction_img(image)