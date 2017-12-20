import os

import numpy as np
import pandas as pd

from keras.utils import to_categorical
from sklearn.model_selection import KFold, train_test_split


def load_data(path):
    train = pd.read_json(os.path.join(path, "./train.json"))
    test = pd.read_json(os.path.join(path, "./test.json"))
    return (train, test)
    

def preprocess(df, means=(-22.159262, -24.953745), stds=(5.33146, 4.5463958)):
    X_band_1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) 
                         for band in df["band_1"]])
    X_band_2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) 
                         for band in df["band_2"]])

    X_band_1 = (X_band_1 - means[0]) / stds[0]
    X_band_2 = (X_band_2 - means[1]) / stds[1]
    
    images = np.concatenate([X_band_1[:, :, :, np.newaxis], 
                             X_band_2[:, :, :, np.newaxis]], 
                            axis=-1)
    return images


def prepare_data_cv(path):
    train, test = load_data(path)
    X_train, y_train = (preprocess(train), 
                        to_categorical(train['is_iceberg'].as_matrix().reshape(-1, 1)))
    
    kfold_data = []
    kf = KFold(n_splits=5, shuffle=True, random_state=0xCAFFE)
    
    for train_indices, val_indices in kf.split(y_train):
        X_train_cv = X_train[train_indices]
        y_train_cv = y_train[train_indices]
        
        X_val = X_train[val_indices]
        y_val = y_train[val_indices]
        
        kfold_data.append((X_train_cv, y_train_cv, X_val, y_val))
    
    X_test = preprocess(test)
    
    return (kfold_data, X_test)


def prepare_data(path):
    train, test = load_data(path)
    X_train, y_train = (preprocess(train), 
                        to_categorical(train['is_iceberg'].as_matrix().reshape(-1, 1)))

    X_train_cv, X_valid, y_train_cv, y_valid = train_test_split(X_train, 
                                                                y_train, 
                                                                random_state=0xCAFFE, 
                                                                train_size=0.75)

    X_test = preprocess(test)

    return ([(X_train_cv, y_train_cv, X_valid, y_valid)], X_test)
