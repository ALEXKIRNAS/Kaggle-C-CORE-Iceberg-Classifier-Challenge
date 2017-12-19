import os

import numpy as np
import pandas as pd

from sklearn.metrics import log_loss

from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau

from data_loader import prepare_data, prepare_data_cv, load_data
from base_model import get_base_model
from data_generation import get_data_generator

def get_model_callbacks(save_dir):
    stopping = EarlyStopping(monitor='val_loss', 
                             min_delta=1e-4, 
                             patience=200, 
                             verbose=False, 
                             mode='min')
    
    board_path = os.path.join(save_dir, 'board')
    if not os.path.exists(board_path):
        os.makedirs(board_path)
    
    board = TensorBoard(log_dir=board_path)
    
    lr_sheduler = ReduceLROnPlateau(monitor='val_loss', 
                                    factor=0.7, 
                                    patience=20, 
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


def load_model(model_loader_fn):
    model = model_loader_fn()
    model.summary()
    return model


def prepare_submission(models_proba, path):
    _, test = load_data('../input')
    proba = np.mean(models_proba, axis=0)
    
    submission = pd.DataFrame()
    submission['id'] = test['id']
    submission['is_iceberg'] = proba.reshape((proba.shape[0]))
    submission.to_csv(path, index=False)
    

def main():
    (kfold_data, X_test) = prepare_data_cv('../input')
    
    models_proba = []
    models_acc = []
    models_logloss = []
    
    for idx, data in enumerate(kfold_data):
        X_train, y_train, X_valid, y_valid = data
        
        model = load_model(get_base_model)
        callbacks = get_model_callbacks(save_dir=('../experiments/base_model_aug/fold_%02d' % idx))
        data_generator = get_data_generator(X_train, y_train, batch_size=256)
        
        model.fit_generator(
            data_generator,
            steps_per_epoch=10,
            epochs=1000,
            verbose=True,
            validation_data=(X_valid, y_valid),
            callbacks=callbacks,
            shuffle=True)

        model.load_weights(filepath=('../experiments/base_model_aug/fold_%02d/model/model_weights.hdf5' % idx))
        score = model.evaluate(X_valid, y_valid, verbose=False)
        proba = model.predict_proba(X_valid)
        
        models_proba.append(model.predict_proba(X_test)[:, 1])
        models_acc.append(score[1])
        models_logloss.append(log_loss(y_valid.argmax(axis=1), proba))
    
    print('Acc:\nMean: %f\nStd: %f\nMin: %f\nMax: %f\n\n' % (np.mean(models_acc), 
                                                             np.std(models_acc),
                                                             np.min(models_acc),
                                                             np.max(models_acc)))

    print('Loss:\nMean: %f\nStd: %f\nMin: %f\nMax: %f\n\n' % (np.mean(models_logloss), 
                                                              np.std(models_logloss),
                                                              np.min(models_logloss),
                                                              np.max(models_logloss)))

    prepare_submission(models_proba, '../submission.csv')
    
    
if __name__ == '__main__':
    main()
