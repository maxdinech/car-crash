"""
gtsrb_loader.py
~~~~~~~~~~~~~~~

Convertit et normalise les images organisés en 43 dossiers de catégories dans
'Final_Training/'' en deux tableaux numpy : 'images' et 'labels'.

Les labels sont des tableaux de 0 avec un 1 à la position de la catégorie du
panneau, les images sont des matrices de 40*40 pixels en nuances de gris (blanc
= 0, noir = 1).
"""

# BIBLIOTHÈQUES EXTERNES
# ----------------------

import os
import numpy as np
import glob
from skimage import io, color, transform
from keras.utils.np_utils import to_categorical


# TRAITEMENT DES DONNÉES
# ----------------------

data_url = 'Final_Training/Images/'


def traite_image(chemin_image):
	image = io.imread(chemin_image)
	image = transform.resize(image, (40, 40), mode='wrap')
	image = color.rgb2gray(image)
	return image


def traite_label(chemin_image):
	return int(chemin_image.split('/')[-2])


def gtsrb(n):
	images = []
	labels = []
	chemins_images = glob.glob(os.path.join(data_url, '*/*.ppm'))
	np.random.shuffle(chemins_images)
	
	for chemin_image, _ in zip(chemins_images, range(n)):
		images.append(traite_image(chemin_image))
		labels.append(traite_label(chemin_image))
	images = np.array(images)
	labels = to_categorical(labels, 43)
	return images, labels