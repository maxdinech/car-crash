'''
GTSRB/kerasCNN.py
~~~~~~~~~~~~~~~~~
'''


# HYPERPARAMÈTRES
# ---------------

couleur = 'clahe'  # 'grey' ou 'clahe'
ext, dist = True, True

epochs = 30
batch_size = 128


# BIBLIOTHÈQUES EXTERNES
# ----------------------

import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D

from keras.models import load_model
from keras.callbacks import TensorBoard


# LECTURE DES DONNÉES
# -------------------

print('\nCouleur : ' + couleur + ', Mode : ext'*ext + '+dist'*dist + '\n')

nom_images = 'images_' + couleur + '_ext'*ext + '_dist'*dist + '.npy'

train_images = np.load('data/train/images_' + couleur + '_ext'*ext + '_dist'*dist + '.npy')
train_labels = np.load('data/train/labels_' + couleur + '_ext'*ext + '_dist'*dist + '.npy')

test_images = np.load('data/test/images_' + couleur + '.npy')
test_labels = np.load('data/test/labels_' + couleur + '.npy')

# Il faut ajouter explicitement la dimension RGB, ici 1
train_images = train_images.reshape(train_images.shape[0], 40, 40, 1)
test_images = test_images.reshape(test_images.shape[0], 40, 40, 1)



# DÉFINITION ET COMPILATION DU MODÈLE
# -----------------------------------

model = Sequential([
    Convolution2D(40, (7,7), input_shape=(40,40,1), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Convolution2D(40, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(100, activation='relu'),
    Dropout(0.4),
    Dense(43, activation='softmax')
])


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    


# ENTRAÎNEMENT DU RÉSEAU
# ----------------------


# tensorboard = TensorBoard(log_dir='./logs',
#                         histogram_freq=10,  # nombre d'hist. par batch
#                         batch_size=batch_size,
#                         write_graph=True,
#                         write_grads=False,
#                         write_images=False)

model.fit(train_images, train_labels,
          shuffle=True,
          batch_size = batch_size,
          epochs = epochs,
          validation_data = (test_images, test_labels))


# PRÉDICTIONS
# -----------

def prediction(n, images=test_images):
    plt.imshow(images[n].reshape(40,40), cmap='gray')
    resultat = model.predict(images[n].reshape(1, 40, 40, 1))
    prediction = np.argmax(resultat)
    proba = resultat[0,prediction]
    plt.title('{} -- {}%'.format(panneau(prediction), (100*proba).round(2)))
    plt.show()

def panneau(n):
    panneaux = np.load('data/noms_panneaux.npy')
    return panneaux[n]

def erreurs(images=test_images, labels=test_labels):
    l = []
    for i in range(len(images)):
        resultat = model.predict(images[i].reshape(1, 40, 40, 1))
        if np.argmax(resultat) != np.argmax(labels[i]):
            l.append(i)
    print("pourcentage d'erreurs :", 100*len(l)/len(images), '%')
    return l

def ascii_print(image):
    image = image.reshape(40,40)
    for ligne in image:
        for pix in ligne:
            print(2*" ░▒▓█"[int(pix*4.999)], end='')
        print('')

def prediction_ascii(n, images=test_images):
    ascii_print(images[n])
    resultat = model.predict(images[n].reshape(1, 40, 40, 1))
    prediction = np.argmax(resultat)
    proba = resultat[0,prediction]
    print('{} -- {}%'.format(panneau(prediction), (100*proba).round(2)))

