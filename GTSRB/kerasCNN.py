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

color = 'clahe'  # 'grey' ou 'clahe'
mode = 'ext'  # 'ext' ou ''
epochs = 12
batch_size = 128


# BIBLIOTHÈQUES EXTERNES
# ----------------------

import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D



# LECTURE DES DONNÉES
# -------------------

print('\nCouleur : ' + color + ', Mode : '*(mode!='') + mode + '\n')

train_images = np.load("data/train/train_" + color + "_"*(mode!='') + mode + ".npy")
train_labels = np.load("data/train/train_labels" + "_"*(mode!='') + mode + ".npy")

val_images = np.load("data/validation/val_" + color + ".npy")
val_labels = np.load("data/validation/val_labels.npy")

# Il faut ajouter explicitement la dimension RGB, ici 1
train_images = train_images.reshape(train_images.shape[0], 40, 40, 1)
val_images = val_images.reshape(val_images.shape[0], 40, 40, 1)



# DÉFINITION ET COMPILATION DU MODÈLE
# -----------------------------------

model = Sequential([
	Convolution2D(32, (5,5), input_shape=(40,40,1), activation='relu'),
	MaxPooling2D(pool_size=(2,2)),
	Dropout(0.2),
	Flatten(),
	Dense(128, activation='relu'),
	Dense(43, activation='softmax')
])

model.compile(loss='categorical_crossentropy',
			  optimizer='adam',
			  metrics=['accuracy'])



# ENTRAÎNEMENT DU RÉSEAU
# ----------------------

history = model.fit(train_images, train_labels,
					batch_size = batch_size,
					epochs = epochs,
					validation_data = (val_images, val_labels))

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Précision - couleur : ' + color + ', mode : '*(mode!='') + mode)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['entraînement', 'validation'], loc='upper left')
plt.show()



# PRÉDICTIONS
# -----------

def prediction(n):
	plt.imshow(images[n].reshape(40,40), cmap='gray')
	resultat = model.predict(images[n].reshape(1, 40, 40, 1))
	prediction = np.argmax(resultat)
	proba = resultat[0,prediction]
	plt.title("{} -- {}%".format(panneau(prediction), (100*proba).round(2)))
	plt.show()

def panneau(n):
	panneaux = np.load("data/noms_panneaux.npy")
	return panneaux[n]

def erreurs(a, b):
	l = []
	for i in range(a, b):
		resultat = model.predict(images[i].reshape(1, 40, 40, 1))
		if np.argmax(resultat) != np.argmax(labels[i]):
			l.append(i)
	print("pourcentage d'erreurs :", 100*len(l)/(b-a), "%")
	return l
