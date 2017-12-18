import os

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, Activation
from keras.layers import GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras import initializers
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from keras.utils import to_categorical


def load_data(path):
    train = pd.read_json(os.path.join(path, "./train.json"))
    test = pd.read_json(os.path.join(path, "./test.json"))
    return (train, test)
    

def preprocess(df):
    X_band_1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) 
                         for band in df["band_1"]])
    X_band_2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) 
                         for band in df["band_2"]])
    
    X_band_1_min = X_band_1.min(axis=(1, 2), keepdims=True)
    X_band_1_max = X_band_1.max(axis=(1, 2), keepdims=True)
    
    X_band_2_min = X_band_2.min(axis=(1, 2), keepdims=True)
    X_band_2_max = X_band_2.max(axis=(1, 2), keepdims=True)
    
    X_band_1 = (X_band_1 - X_band_1_min) / (X_band_1_max - X_band_1_min) - 0.5
    X_band_2 = (X_band_2 - X_band_2_min) / (X_band_2_max - X_band_2_min) - 0.5
    
    images = np.concatenate([X_band_1[:, :, :, np.newaxis], 
                             X_band_2[:, :, :, np.newaxis]], 
                            axis=-1)
    return images


def prepare_data(path):
    train, test = load_data(path)
    X_train, y_train = (preprocess(train), 
                        to_categorical(train['is_iceberg'].as_matrix().reshape(-1, 1)))
    
    X_train_cv, X_valid, y_train_cv, y_valid = train_test_split(X_train, 
                                                                y_train, 
                                                                random_state=0xCAFFE, 
                                                                train_size=0.75)
    
    X_test = preprocess(test)
    
    return (X_train_cv, y_train_cv, X_valid, y_valid, X_test)


def get_base_model():
    #Building the model
    model=Sequential()
    
    #Conv Layer 1
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(75, 75, 2)))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    #Conv Layer 2
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu' ))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    #Conv Layer 3
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    #Conv Layer 4
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    #Flatten the data for upcoming dense layers
    model.add(Flatten())

    #Dense Layers
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.7))

    #Dense Layer 2
    model.add(Dense(128))
    model.add(Activation('relu'))

    #Sigmoid Layer
    model.add(Dense(2))
    model.add(Activation('softmax'))

    opt=SGD(lr=0.01, momentum=0.9)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    
    return model


def get_model_callbacks(save_dir):
    stopping = EarlyStopping(monitor='val_loss', 
                             min_delta=1e-4, 
                             patience=50, 
                             verbose=False, 
                             mode='min')
    
    board_path = os.path.join(save_dir, 'board')
    if not os.path.exists(board_path):
        os.makedirs(board_path)
    
    board = TensorBoard(log_dir=board_path)
    
    lr_sheduler = ReduceLROnPlateau(monitor='val_loss', 
                                    factor=0.9, 
                                    patience=10, 
                                    verbose=True, 
                                    mode='min', 
                                    epsilon=1e-4,
                                    min_lr=1e-5)
    
    model_path = os.path.join(save_dir, 'model/model_weights.hdf5')
    if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))
    
    model_checkpoint = ModelCheckpoint(model_path, 
                                       monitor='val_loss', 
                                       verbose=False,
                                       save_best_only=True, 
                                       save_weights_only=False, 
                                       mode='min', 
                                       period=1)
    
    callbacks = [stopping, board, lr_sheduler, model_checkpoint]
    return callbacks


def load_model():
    model = get_base_model()
    model.summary()
    return model


def prepare_submission(model, X_test, path):
    predicted_test = model.predict_proba(X_test)
    
    _, test = load_data('../input')
    submission = pd.DataFrame()
    submission['id'] = test['id']
    submission['is_iceberg'] = predicted_test[:, 1].reshape((predicted_test.shape[0]))
    submission.to_csv(path, index=False)
    

def main():
    (X_train, y_train, X_valid, y_valid, X_test) = prepare_data('../input')
    model = load_model()
    callbacks = get_model_callbacks(save_dir='../experiments/base_model')
    
    model.fit(X_train, y_train,
              batch_size=128,
              epochs=1000,
              verbose=1,
              validation_data=(X_valid, y_valid),
              callbacks=callbacks)
    
    model.load_weights(filepath='../experiments/base_model/model/model_weights.hdf5')
    score = model.evaluate(X_valid, y_valid, verbose=1)
    print('Val loss:', score[0])
    print('Val accuracy:', score[1])
    
    prepare_submission(model, X_test, '../submission.csv')
    
    
if __name__ == '__main__':
    main()
