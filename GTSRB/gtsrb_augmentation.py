'''
gtsrb_loader.py
~~~~~~~~~~~~~~~

Charge et augmente une base de données
'''

# BIBLIOTHÈQUES EXTERNES
# ----------------------

import numpy as np
import random
from skimage import transform
from joblib import Parallel, delayed
import matplotlib.pyplot as plt


# ROTATIONS ET SYMÉTRIES
# ----------------------

auto_miroir_vertical = [11, 12, 13, 15, 17, 18, 22, 26, 30, 35]
auto_miroir_horizontal = [1, 5, 12, 15, 17]
auto_miroir_double = [12, 15, 17, 32]

mutuel_miroir_vert = [[19,20], [33,34], [36,37], [38,39], [20,19], [34,33], [37,36], [39,38]]


# AUGMENTATION
# ------------

def symétries(images, labels):
    labels_num = np.array([np.argmax(i) for i in labels])
    images_ext = np.empty((0, 40, 40), dtype = images.dtype)
    labels_ext = np.empty((0, 43), dtype = labels.dtype)
    for c in range(43):
        print(c)
        images_c = images[labels_num == c]
        images_ext = np.append(images_ext, images_c, axis=0)
        if c in auto_miroir_horizontal:
            images_ext = np.append(images_ext, images_c[:, :, ::-1], axis=0)
        if c in auto_miroir_vertical:
            images_ext = np.append(images_ext, images_c[:, ::-1, :], axis=0)
        if c in auto_miroir_double:
            images_ext = np.append(images_ext, images_c[:, ::-1, ::-1], axis=0)
        nouv_labels = np.full((len(images_ext) - len(labels_ext), 43), np.eye(43)[c])
        labels_ext = np.append(labels_ext, nouv_labels, axis=0)
        if [c, c+1] in mutuel_miroir_vert:
            images_ext = np.append(images_ext, images_c[:, ::-1, :], axis=0)
            nouv_labels = np.full((len(images_ext) - len(labels_ext), 43), np.eye(43)[c+1])
            labels_ext = np.append(labels_ext, nouv_labels, axis=0)
        if [c, c-1] in mutuel_miroir_vert:
            images_ext = np.append(images_ext, images_c[:, ::-1, :], axis=0)
            nouv_labels = np.full((len(images_ext) - len(labels_ext), 43), np.eye(43)[c-1])
            labels_ext = np.append(labels_ext, nouv_labels, axis=0)
    return images_ext, labels_ext


# Multiplie par n la taille de la BDD
def distorsions(images, labels, n=5):
    labels_ext = labels
    images_ext = images
    for i in range(n-1):
        labels_ext = np.append(labels_ext, labels, axis=0)
    for i in range(n-1):
        images_ext_i = Parallel(n_jobs=16)(delayed(distorsion)(img) for img in images)
        images_ext = np.append(images_ext, np.array(images_ext_i), axis=0)
    rng_state = np.random.get_state()
    np.random.shuffle(images_ext)
    np.random.set_state(rng_state)
    np.random.shuffle(labels_ext)
    return images_ext, labels_ext


def distorsion(image):
    src = np.array([[0, 0], [0, 40], [40, 40], [40, 0]])
    dst = np.array([[random.randrange(-6, 6), random.randrange(-6, 6)],
                    [random.randrange(-6, 6), 40 - random.randrange(-6, 6)],
                    [40 - random.randrange(-6, 6),40 - random.randrange(-6, 6)],
                    [40 - random.randrange(-6, 6), random.randrange(-6, 6)]])
    disto = transform.ProjectiveTransform()
    disto.estimate(src, dst)
    image = transform.warp(image, disto, output_shape=(40, 40), mode='edge')
    return image

def transforme(image):
    src = np.array([[0, 0], [0, 40], [40, 40], [40, 0]])
    dst = np.array([[random.randrange(-5, 5), random.randrange(-5, 5)],
                    [random.randrange(-5, 5), 40 - random.randrange(-5, 5)],
                    [40 - random.randrange(-5, 5),40 - random.randrange(-5, 5)],
                    [40 - random.randrange(-5, 5), random.randrange(-5, 5)]])
    disto = transform.ProjectiveTransform()
    disto.estimate(src, dst)
    plt.imshow(image, cmap='gray')
    plt.show()
    plt.imshow(transform.warp(image, disto, output_shape=(40, 40), mode='edge'), cmap='gray')
    plt.show()


# Mélanges synchronisés de deux listes
# rng_state = np.random.get_state()
# np.random.shuffle(images_ext)
# np.random.set_state(rng_state)
# np.random.shuffle(labels_ext)