"""
gtsrb_loader.py
~~~~~~~~~~~~~~~

Convertit et normalise les images organisés en 43 dossiers de catégories dans
'Final_Training/'' en deux tableaux numpy : 'images' et 'labels'.

Les images sont enregistrées en 40x40 piexls

"""


color = 'clahe'  # 'rgb', 'grey' ou 'clahe'


# BIBLIOTHÈQUES EXTERNES
# ----------------------

import os
import numpy as np
import glob
from skimage import io, color, transform, exposure


# TRAITEMENT DES DONNÉES
# ----------------------

data_url = 'data/Training/'


def traite_image(chemin_image):
	# Lecture de l'image
	image = io.imread(chemin_image)
	# Redimensionnement en 40x40 pixels
	image = transform.resize(image, (40, 40), mode='wrap')
	# Ajustement local de l'exposition
	if color == 'clahe':
		image = exposure.equalize_adapthist(image)
	# Conversion en nuances de gris
	if not color == 'rgb':
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
	labels = np.eye(43)[np.array(labels)]  # Conversion entiers -> catégories
	return images, labels


def save_gtsrb(images, labels):
	