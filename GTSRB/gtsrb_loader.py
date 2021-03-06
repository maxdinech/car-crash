"""
Charge la base de données GTSRB


TODO: Enregistrer au format Bytes et diviser par 255 à l'appel (comme MNIST)
"""



import os
import wget
import shutil
import zipfile
import numpy as np
import pandas as pd
import glob
import torch
from skimage import io, color, transform, exposure
from joblib import Parallel, delayed


# Utilise automatiquement le GPU si CUDA est disponible

if torch.cuda.is_available():
    ltype = torch.cuda.LongTensor
else:
    ltype = torch.LongTensor


# Téléchargement et décompression des images

def get_train_folder():
    train_url = 'http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip'
    if not os.path.exists('data'):
        os.makedirs('data')
    if not os.path.exists('data/Final_Training'):
        print("Downloading the train database...")
        wget.download(train_url, 'data/train.zip')
        print("\nDownload complete.")
        print("Unzipping the train database...")
        zip_ref = zipfile.ZipFile('data/train.zip', 'r')
        zip_ref.extractall('data/')
        zip_ref.close()
        print("Unzip complete.")
        shutil.move('data/GTSRB/Final_Training', 'data/')
        shutil.rmtree('data/GTSRB')
        os.remove('data/train.zip')

def get_test_folder():
    test_url = 'http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_Images.zip'
    test_labels_url = 'http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_GT.zip'
    if not os.path.exists('data'):
        os.makedirs('data')
    if not os.path.exists('data/Final_Test'):
        print("Downloading the test database...")
        wget.download(test_url, 'data/test.zip')
        wget.download(test_labels_url, 'data/test_labels.zip')
        print("\nDownload complete.")
        print("Unzipping the test database...")
        zip_ref = zipfile.ZipFile('data/test.zip', 'r')
        zip_ref.extractall('data/')
        zip_ref = zipfile.ZipFile('data/test_labels.zip', 'r')
        zip_ref.extractall('data/GTSRB/Final_Test')
        zip_ref.close()
        print("Unzip complete.")
        shutil.move('data/GTSRB/Final_Test', 'data/')
        shutil.rmtree('data/GTSRB')
        os.remove('data/test.zip')
        # Réorganisation du dossier Images/ en sous-dossiers
        data_url = 'data/Final_Test/Images/'
        labels = np.array(pd.read_csv('data/Final_Test/GT-final_test.csv', sep=';'))[:,7]
        for i in range(0, 43):
            if not os.path.exists(data_url + str(i).zfill(5)):
                os.makedirs(data_url + str(i).zfill(5))
        for i in range(len(labels)):
            image = str(i).zfill(5) + '.ppm'
            label = str(labels[i]).zfill(5)
            os.rename(data_url + image, data_url + label + '/' + image)


# Transformation en tenseur

def traite_image(chemin_image, couleur):
    # Lecture de l'image
    image = io.imread(chemin_image)
    # Redimensionnement en 40x40 pixels
    image = transform.resize(image, (40, 40), mode='wrap')
    # Ajustement local de l'exposition
    if couleur == 'clahe':
        image = exposure.equalize_adapthist(image)
    # Conversion en nuances de gris
    if not couleur == 'rgb':
        image = color.rgb2gray(image)
    return image

def traite_label(chemin_image):
    return int(chemin_image.split('/')[-2])


# Enregistrement des tenseurs

def save_train(images, labels, couleur):
    if not os.path.exists('data/' + couleur):
        os.makedirs('data/' + couleur)
    torch.save((images, labels), 'data/' + couleur + '/train.pt')

def save_test(images, labels, couleur):
    if not os.path.exists('data/' + couleur):
        os.makedirs('data/' + couleur)
    torch.save((images, labels), 'data/' + couleur + '/test.pt')


# Chargement des tenseurs 

def train(couleur, nb_train=39209):  # Couleur : 'rgb', 'grey', 'clahe'
    if not os.path.exists('data/' + couleur + '/train.pt'):
        get_train_folder()
        chemins_images = glob.glob(os.path.join('data/Final_Training/Images/', '*/*.ppm'))
        images = Parallel(n_jobs=16)(delayed(traite_image)(path, couleur) for path in chemins_images)
        labels = Parallel(n_jobs=16)(delayed(traite_label)(path) for path in chemins_images)
        images = torch.Tensor(images)
        labels = torch.Tensor(labels).type(ltype)
        if couleur == 'rgb':
            images = images.permute(0, 3, 1, 2)
        else:
            images = images.view(nb_train, 1, 40, 40)
        save_train(images, labels, couleur)
    images, labels = torch.load('data/' + couleur + '/train.pt')
    images, labels = images[:nb_train], labels[:nb_train]
    return images, labels


def test(couleur, nb_val=12630):
    if not os.path.exists('data/' + couleur + '/test.pt'):
        get_test_folder()
        chemins_images = glob.glob(os.path.join('data/Final_Test/Images/', '*/*.ppm'))
        images = Parallel(n_jobs=16)(delayed(traite_image)(path, couleur) for path in chemins_images)
        labels = Parallel(n_jobs=16)(delayed(traite_label)(path) for path in chemins_images)
        images = torch.Tensor(images)
        labels = torch.Tensor(labels).type(ltype)
        if couleur == 'rgb':
            images = images.permute(0, 3, 1, 2)
        else:
            images = images.view(nb_val, 1, 40, 40)
        save_test(images, labels, couleur)
    images, labels = torch.load('data/' + couleur + '/test.pt')
    images, labels = images[:nb_val], labels[:nb_val]
    return images, labels
