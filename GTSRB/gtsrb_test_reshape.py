"""
gtsrb_loader.py
~~~~~~~~~~~~~~~


"""


import os
import numpy as np
import pandas as pd
import glob


# TRAITEMENT DES DONNÉES
# ----------------------

data_url = 'data/Final_Test/Images/'

labels = np.array(pd.read_csv('data/Final_Test/GT-final_test.csv', sep=';'))[:,7]

def reshape():
	for i in range(0, 43):
		os.makedirs(data_url + str(i).zfill(5))
	for i in range(len(labels)):
		os.rename(data_url + str(i).zfill(5) + '.ppm', data_url + str(labels[i]).zfill(5) + '/')
