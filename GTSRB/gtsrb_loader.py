"""
gtsrb_loader.py
~~~~~~~~~~~~~~~


"""


color = 'grey'  # 'rgb', 'grey' ou 'clahe'

source = "training"  # 'training' ou 'test'

# BIBLIOTHÈQUES EXTERNES
# ----------------------

import os
import numpy as np
import glob
from skimage import io, color, transform, exposure


# TRAITEMENT DES DONNÉES
# ----------------------

data_url = 'data/Final_T' + source[1:] + '/Images/'



def traite_image(chemin_image):
	# Lecture de l'image
	image = io.imread(chemin_image)
	# Redimensionnement en 40x40 pixels
	image = transform.resize(image, (40, 40), mode='wrap')
	# Ajustement local de l'exposition
	if color == 'clahe':
		image = exposure.equalize_adapthist(image)
	# Conversion en nuances de gris
	image = color.rgb2gray(image)
	return image


def traite_label(chemin_image):
	return int(chemin_image.split('/')[-2])


def gtsrb(n):
	images = []
	labels = []
	chemins_images = glob.glob(os.path.join(data_url, '*/*.ppm'))
	for chemin_image, _ in zip(chemins_images, range(n)):
		images.append(traite_image(chemin_image))
		labels.append(traite_label(chemin_image))
	images = np.array(images)
	labels = np.eye(43)[np.array(labels)]  # Conversion entiers -> catégories
	return images, labels


def save(images, labels):
	np.save('data/' + source + '/images_' + color, images)
	np.save('data/' + source + '/labels_' + color, labels)
