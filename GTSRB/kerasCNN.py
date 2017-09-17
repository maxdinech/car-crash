"""
GTSRB/kerasCNN.py
~~~~~~~~~~~~~~~~~

Résultats attendus : environ 95% de succès.
"""

# BIBLIOTHÈQUES EXTERNES
# ----------------------

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D

import gtsrb_loader


# BASE DE DONNÉES
# ---------------

images, labels = gtsrb_loader.gtsrb(12000)

# Il faut ajouter explicitement la dimension RGB, ici 1
images = images.reshape(images.shape[0], 40, 40, 1)


# DÉFINITION ET COMPILATION DU MODÈLE
# -----------------------------------

model = Sequential()

model.add(Convolution2D(32, (5,5), input_shape=(40,40,1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(43, activation='softmax'))

model.compile(loss='categorical_crossentropy',
			  optimizer='adam',
			  metrics=['accuracy'])


# ENTRAÎNEMENT DU RÉSEAU
# ----------------------

history = model.fit(images, labels,
					batch_size=128,
					epochs=20,
					validation_split=1/6)
