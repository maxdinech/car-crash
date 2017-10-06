"""
mnist_loader.py
~~~~~~~~~~~~~~~

Convertit la base de données 'train.csv' en deux tableaux numpy : 'images' et
'labels'.

Les labels sont des tableaux de 0 avec un 1 à la position du chiffre, les images
sont des matrices de 28*28 pixels en nuances de gris (blanc = 0, noir = 1).
"""

# BIBLIOTHÈQUES EXTERNES
# ----------------------

import numpy as np
import pandas as pd

mode = 'train'  # 'train' ou 'test'


# TRAITEMENT DES DONNÉES
# ----------------------

data_url = 'data/' + mode + '.csv'


def mnist():
    data = pd.read_csv(data_url, header=None)
    images = ((data.ix[:,1:]).values).astype('float32') / 255.
    images = images.reshape(images.shape[0], 28, 28)
    labels = np.eye(10)[data.ix[:,0].values.astype('int32')]
    return images, labels


def save_mnist(images, labels):
    np.save("data/" + mode + "_images", images)
    np.save("data/" + mode + "_labels", labels)
