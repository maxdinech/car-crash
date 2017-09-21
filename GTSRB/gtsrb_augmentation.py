"""
gtsrb_loader.py
~~~~~~~~~~~~~~~

Charge et augmente une base de données
"""


mode = 'clahe'  # 'rgb', 'grey' ou 'clahe'


# BIBLIOTHÈQUES EXTERNES
# ----------------------

import numpy as np
from skimage import transform

import matplotlib.pyplot as plt


# LECTURE DES DONNÉES
# -------------------


images = np.load("data/train_" + mode + ".npy")
labels = np.load("data/train_labels.npy")


# ROTATIONS ET SYMÉTRIES
# ----------------------

auto_miroir_vertical = [11, 12, 13, 15, 17, 18, 22, 26, 30, 35]
auto_miroir_horizontal = [1, 5, 12, 15, 17]
auto_miroir_double = [12, 15, 17, 32]

mutuel_miroir_vert = [[19,20], [33,34], [36,37], [38,39], [20,19], [34,33], [37,36], [39,38]]


# AUGMENTATION
# ------------



def augmente(images, labels):
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



images_ext, labels_ext = augmente(images, labels)

np.save("data/images_" + mode + "_ext", images_ext)
np.save("data/labels_ext", labels_ext)