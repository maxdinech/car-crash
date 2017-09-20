"""
MNIST/kerasMLP.py
~~~~~~~~~~~~~~~~~
"""

# BIBLIOTHÈQUES EXTERNES
# ----------------------

import numpy as np

from keras.models import Sequential
from keras.layers import Reshape, Dense, Activation
from keras.optimizers import RMSprop

from skimage import io, color

import mnist_loader


# BASE DE DONNÉES
# ---------------

images = np.load("data.images.npy")
labels = np.load("data.labels.npy")


# DÉFINITION ET COMPILATION DU RÉSEAU
# -----------------------------------

model = Sequential()

model.add(Reshape((28*28,), input_shape=(28,28)))
model.add(Dense(32, activation='relu', input_dim=(28*28)))
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer=RMSprop(lr=0.001),
			  loss='categorical_crossentropy',
			  metrics=['accuracy'])



# ENTRAÎNEMENT DU RÉSEAU
# ----------------------

history = model.fit(images, labels,
					batch_size=64,
					epochs=25,
					validation_split=1/6)


# PRÉDICTION D'UN CHIFFRE
# -----------------------

def prediction(model):
	image = 1 - color.rgb2grey(io.imread('digit.png'))
	resultat = model.predict(image.reshape(1, 28, 28))
	prediction = np.argmax(resultat)
	proba = resultat[0,prediction]
	return (prediction, proba)
