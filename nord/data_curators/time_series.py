import os
import urllib
import zipfile

import numpy as np
import pandas as pd
import torch
from scipy.io import arff
from sklearn.preprocessing import LabelBinarizer
from torch.utils.data import DataLoader, TensorDataset


def get_activity_recognition_data(train_batch=128, test_batch=128,
                                  differentiate=True, lag_window=52,
                                  lag_overlap_samples=26,
                                  test_subjects=4):

    root = './data/activity_recognition_data'
    zip_file_path = root+'/Activity Recognition from Single Chest-Mounted Accelerometer.zip'
    csv_file_path = zip_file_path.replace('.zip', '')
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00287/Activity%20Recognition%20from%20Single%20Chest-Mounted%20Accelerometer.zip'

    if not os.path.exists(root):
        os.makedirs(root)

    if not os.path.isfile(zip_file_path):
        print('Downloading Activity Recognition Data.')
        urllib.request.urlretrieve(url, zip_file_path)

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(root)

    lags = range(0, lag_window)
    lb = LabelBinarizer().fit([x for x in range(1, 8)])

    def get_subjects(files):
        all_data = []
        all_labels = []
        all_activities = []
        for file in files:
            if file.split('.')[-1] == 'csv':
                df = pd.read_csv(csv_file_path+'/'+file, header=None,
                                 names=['x', 'y', 'z', 'activity'], usecols=[1, 2, 3, 4])
                print(len(df))
                # Last entry is sometimes 0
                df = df.iloc[:-1]
                if differentiate:
                    df[['x', 'y', 'z']] = df[['x', 'y', 'z']].diff()

                df = df.assign(**{
                    '{} (t-{})'.format(col, t): df[col].shift(t)
                    for col in ['x', 'y', 'z']
                    for t in lags

                })

                df.dropna(inplace=True)
                df = df.iloc[::lag_overlap_samples, :]
                data = df.drop(
                    labels=['activity', 'x', 'y', 'z'], axis=1).values
                labels = lb.transform(df['activity'])
                all_data.extend(np.array(data.reshape(-1, 3, lag_window)))
                all_labels.extend(labels)
        return torch.Tensor(all_data), torch.Tensor(all_labels)

    all_files = os.listdir(csv_file_path)

    trainset = TensorDataset(*get_subjects(all_files[:-test_subjects]))
    testset = TensorDataset(*get_subjects(all_files[-test_subjects:]))

    trainloader = DataLoader(trainset, batch_size=train_batch, shuffle=True)
    testloader = DataLoader(testset, batch_size=test_batch, shuffle=True)

    return trainloader, testloader, 7


def get_earthquake_data(train_batch=128, test_batch=128):

    root = './data/earthquake_data'
    zip_file_path = root+'/Earthquakes.zip'

    def get_dataset(file):
        data = arff.loadarff(root+'/Earthquakes_TRAIN.arff')
        data = pd.DataFrame(data[0])
        labels = torch.Tensor(data['target'].astype(int).values)
        labels = labels.unsqueeze(1)
        data = torch.Tensor(data.drop(labels=['target'], axis=1).values)
        data = data.unsqueeze(1)
        dataset = TensorDataset(data, labels)
        return dataset

    url = 'https://timeseriesclassification.com/Downloads/Earthquakes.zip'

    if not os.path.exists(root):
        os.makedirs(root)

    if not os.path.isfile(zip_file_path):
        print('Downloading Earthquake Data.')
        urllib.request.urlretrieve(url, zip_file_path)

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(root)

    trainset = get_dataset(root+'/Earthquakes_TRAIN.arff')
    testset = get_dataset(root+'/Earthquakes_TEST.arff')

    trainloader = DataLoader(trainset, batch_size=train_batch, shuffle=True)
    testloader = DataLoader(testset, batch_size=test_batch, shuffle=True)

    return trainloader, testloader, 2

