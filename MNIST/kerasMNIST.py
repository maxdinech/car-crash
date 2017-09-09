"""
kerasMNIST.py
~~~~~~~~~~~~~
"""

# BIBLIOTHÈQUES EXTERNES
# ----------------------

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop

from keras.utils.np_utils import to_categorical


# TRAITEMENT DES DONNÉES
# ----------------------

data_url = '/home/max/Dropbox/Prépa/TIPE/car-crash/MNIST/train.csv'
data = pd.read_csv(data_url, header=None)

images = ((data.ix[:,1:]).values).astype('float32') / 255.
labels = to_categorical(data.ix[:,0].values.astype('int32'), 10)
# to_categorical convertit n en un tableau tq t[n]=1, t[i] = 0 sinon


# DÉFINITION ET COMPILATION DU RÉSEAU
# -----------------------------------

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])



# ENTRAÎNEMENT DU RÉSEAU
# ----------------------

history = model.fit(images, labels, validation_split=1./6., epochs=20, batch_size=128)

