"""
GTSRB/kerasCNN.py
~~~~~~~~~~~~~~~~~

Résultats attendus : environ 90% de succès.
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

images, labels = gtsrb_loader.gtsrb(11000)

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

def entrainement():
	history = model.fit(images, labels,
						batch_size=128,
						epochs=20,
						validation_split=1/11)
	return history


# PRÉDICTIONS
# -----------

def prediction(n):
	plt.imshow(images[n].reshape(40,40), cmap='gray')
	resultat = model.predict(images[n].reshape(1, 40, 40, 1))
	prediction = np.argmax(resultat)
	proba = resultat[0,prediction]
	if prediction == np.argmax(labels[n]):
		color = 'green'
	else:
		color = 'red'
	plt.title("{} -- {}%".format(panneau(prediction), (100*proba).round(2)), color = color)
	plt.show()

def panneau(n):
	panneaux = ["Limitation de vitesse : 20km/h", "Limitation de vitesse : 30km/h", "Limitation de vitesse : 50km/h", "Limitation de vitesse : 60km/h", "Limitation de vitesse : 70km/h", "Limitation de vitesse : 80km/h", "Fin de limitation de vitesse : 80km/h", "Limitation de vitesse : 100km/", "Limitation de vitesse : 120km/", "Interdiction de dépasser", "Interdiction de dépasser (poids lourds)", "Priorité à la prochaine intersection", "Route prioritaire", "Cédez le passage", "Stop", "Circulation interdite", "Circulation interdite aux poids lourds", "Sens interdit", "Danger", "Virage dangereux à gauche", "Virage dangereux à droite", "Succession de virages dangereux", "Dos-d'âne", "Chaussée glissante", "Chaussée rétrécie par la droite", "Travaux", "Feux tricolores", "Traversée de piétons", "Traversée d'enfants", "Traversée de vélos", "Danger : glace/neige", "Traversée d'animaux sauvages", "Fin de toutes limitations", "Tournez à droite", "Tournez à gauche", "Tout droit uniquement", "Tout droit ou à droite", "Tout droit ou à gauche", "Restez à droite", "Restez à gauche", "Giratoire", "Fin d'interdiction de dépasser", "Fin d'interdiction de dépasser (poids lourds)"]
	return panneaux[n]

def erreurs(a, b):
	l = []
	for i in range(a, b):
		resultat = model.predict(images[i].reshape(1, 40, 40, 1))
		if np.argmax(resultat) != np.argmax(labels[i]):
			l.append(i)
	print("pourcentage d'erreurs :", 100*len(l)/(b-a), "%")
	return l
