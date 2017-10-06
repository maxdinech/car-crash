"""Charge la base de données GTSRB"""


couleur = 'clahe'  # 'rgb', 'grey' ou 'clahe'

source = 'train'  # 'train' ou 'test'

# BIBLIOTHÈQUES EXTERNES
# ----------------------

import os
import numpy as np
import glob
from skimage import io, color, transform, exposure
from joblib import Parallel, delayed


# TRAITEMENT DES DONNÉES
# ----------------------

if source == 'test':
    data_url = 'data/Final_Test/Images/'

if source == 'train':
    data_url = 'data/Final_Training/Images/'



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


def gtsrb():
    chemins_images = glob.glob(os.path.join(data_url, '*/*.ppm'))
    images = Parallel(n_jobs=16)(delayed(traite_image)(path) for path in chemins_images)
    labels = Parallel(n_jobs=16)(delayed(traite_label)(path) for path in chemins_images)
    images = np.array(images)
    labels = np.eye(43)[np.array(labels, dtype=int)]  # Conversion entiers -> catégories
    return images, labels


def save(images, labels):
    os.makedirs('data' + source)
    if source == 'test':
        np.save('data/test/images_' + couleur, images)
        np.save('data/test/labels_' + couleur, labels)
    if source == 'train':
        np.save('data/train/images_' + couleur, images)
        np.save('data/train/labels_' + couleur, labels)
