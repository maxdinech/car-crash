"""
GTSRB/kerasCNN.py
~~~~~~~~~~~~~~~~~

Résultats attendus : environ 98% de succès.


Description des données : dossier 'data/'

La base de donnée initiale est constituée de 39203 images de tailles variables.
Ces images ont été mélangées aléatoirement puis normalisées en taille 40x40 px.

Les 3000 dernières images (val_rgb) servent exclusivement à la validation (test
de performance) du réseau.

Les autres images peuvent être utilisées à volonté.

Ces images ont été transformées (_grey ou _clahe) et étendues ou non (_ext) pour
expérimenter la performance du réseau sous dofférents paramètres.
"""


# HYPERPARAMÈTRES
# ---------------

couleur = 'clahe'  # 'grey' ou 'clahe'
ext, dist = False, False

epochs = 15
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

train_images = np.load("data/train/images_" + couleur + "_ext"*ext + "_dist"*dist + ".npy")
train_labels = np.load("data/train/images_" + couleur + "_ext"*ext + "_dist"*dist + ".npy")

test_images = np.load("data/test/images_" + couleur + ".npy")
test_labels = np.load("data/test/labels_" + couleur + ".npy")

# Il faut ajouter explicitement la dimension RGB, ici 1
train_images = train_images.reshape(train_images.shape[0], 40, 40, 1)
test_images = test_images.reshape(test_images.shape[0], 40, 40, 1)



# DÉFINITION ET COMPILATION DU MODÈLE
# -----------------------------------

model = Sequential([
	Convolution2D(20, (5,5), input_shape=(40,40,1), activation='relu'),
	MaxPooling2D(pool_size=(2,2)),
	Convolution2D(40, (5,5), activation='relu'),
	MaxPooling2D(pool_size=(2,2)),
	Dropout(0.4),
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
# 						  histogram_freq=10,  # nombre d'hist. par batch
# 						  batch_size=batch_size,
# 						  write_graph=True,
# 						  write_grads=False,
# 						  write_images=False)

model.fit(train_images, train_labels,
		  shuffle=True,
		  batch_size = batch_size,
		  epochs = epochs,
		  validation_data = (test_images, val_labels))


# PRÉDICTIONS
# -----------

def prediction(n):
	plt.imshow(test_images[n].reshape(40,40), cmap='gray')
	resultat = model.predict(test_images[n].reshape(1, 40, 40, 1))
	prediction = np.argmax(resultat)
	proba = resultat[0,prediction]
	plt.title("{} -- {}%".format(panneau(prediction), (100*proba).round(2)))
	plt.show()

def panneau(n):
	panneaux = np.load("data/noms_panneaux.npy")
	return panneaux[n]

def erreurs():
	l = []
	for i in range(len(test_images)):
		resultat = model.predict(test_images[i].reshape(1, 40, 40, 1))
		if np.argmax(resultat) != np.argmax(val_labels[i]):
			l.append(i)
	print("pourcentage d'erreurs :", 100*len(l)/len(test_images), "%")
	return l
