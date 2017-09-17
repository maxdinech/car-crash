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