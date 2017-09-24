"""
gtsrb_loader.py
~~~~~~~~~~~~~~~


"""


couleur = 'grey'  # 'rgb', 'grey' ou 'clahe'

source = "test"  # 'train' ou 'test'

# BIBLIOTHÈQUES EXTERNES
# ----------------------

import os
import numpy as np
import glob
from skimage import io, color, transform, exposure


# TRAITEMENT DES DONNÉES
# ----------------------

if source == 'test':
	data_url = 'data/Final_Test' + source[1:] + '/Images/'
if source == 'train':
	data_url = 'data/Final_Training' + source[1:] + '/Images/'



def traite_image(chemin_image):
	# Lecture de l'image
	image = io.imread(chemin_image)
	# Redimensionnement en 40x40 pixels
	image = transform.resize(image, (40, 40), mode='wrap')
	# Ajustement local de l'exposition
	if couleur == 'clahe':
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
	if source == 'test':
		np.save('data/test/images_' + couleur, images)
		np.save('data/test/labels_' + couleur, labels)
	if source == 'train':
		np.save('data/train/images_' + couleur, images)
		np.save('data/train/labels_' + couleur, labels)
