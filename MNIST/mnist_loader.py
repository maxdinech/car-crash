# BIBLIOTHÈQUES EXTERNES
# ----------------------

import numpy as np
import pandas as pd
from keras.utils.np_utils import to_categorical


# TRAITEMENT DES DONNÉES
# ----------------------

data_url = 'train.csv'


def mnist():
	data = pd.read_csv(data_url, header=None)
	images = ((data.ix[:,1:]).values).astype('float32') / 255.
	images = images.reshape(images.shape[0], 28, 28)
	labels = to_categorical(data.ix[:,0].values.astype('int32'), 10)
	return images, labels
