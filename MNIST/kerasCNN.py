"""
MNIST/kerasCNN.py
~~~~~~~~~~~~~~~~~
"""

# BIBLIOTHÈQUES EXTERNES
# ----------------------

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D

from skimage import io, color


# BASE DE DONNÉES
# ---------------

images = np.load("data.images.npy")
labels = np.load("data.labels.npy")

# Il faut ajouter explicitement la dimension RGB, ici 1
images = images.reshape(images.shape[0], 28, 28, 1)


# DÉFINITION ET COMPILATION DU MODÈLE
# -----------------------------------

model = Sequential()

model.add(Convolution2D(32, (5,5), input_shape=(28,28,1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
			  optimizer='adam',
			  metrics=['accuracy'])


# ENTRAÎNEMENT DU RÉSEAU
# ----------------------

history = model.fit(images, labels,
					batch_size=128,
					epochs=12,
					validation_split=1/6)


# PRÉDICTION D'UN CHIFFRE
# -----------------------

def prediction(model):
	image = 1 - color.rgb2grey(io.imread('digit.png'))
	resultat = model.predict(image.reshape(1, 28, 28, 1))
	prediction = np.argmax(resultat)
	proba = resultat[0,prediction]
	return (prediction, proba)
