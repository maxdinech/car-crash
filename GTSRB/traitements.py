#!

"""Script d'automatisation des manipulations sur GTSRB"""

import os
import wget
import zipfile

# import gtsrb_augmentation as aug
# import gtsrb_test_reshape as test_reshape


def get_train_folder():
    train_url = 'http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip'
    if not os.path.exists('data'):
        os.makedirs('data')
    if not os.path.exists('data/Final_Training'):
        print("Downloading the train database...")
        wget.download(train_url, 'data/train.zip')
        print("Download complete.")
        print("Unzipping the train database...")
        zip_ref = zipfile.ZipFile('data/train.zip', 'r')
        zip_ref.extractall('data/')
        zip_ref.close()
        print("Unzip complete.")


def get_test_folder():
    test_url = 'http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_Images.zip'
    if not os.path.exists('data'):
        os.makedirs('data')
    if not os.path.exists('data/Final_Test'):
        print("Downloading the test database...")
        wget.download(test_url, 'data/test.zip')
        print("Download complete.")
        print("Unzipping the test database...")
        zip_ref = zipfile.ZipFile('data/test.zip', 'r')
        zip_ref.extractall('data/')
        zip_ref.close()
        print("Unzip complete.")
