import os

import numpy as np
import pandas as pd

from sklearn.metrics import log_loss, roc_auc_score

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


def get_resnext():
    from keras.optimizers import Adam, SGD, RMSprop
    from resnext import ResNext
    model= ResNext(
        input_shape=(75, 75, 3), 
        depth=11, 
        cardinality=4, 
        width=4,
        weight_decay=5e-4,
        include_top=True, 
        weights=None,
        classes=2)
    
    opt=SGD(lr=0.03, momentum=0.9)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    
    return model


def get_nasnet():
    from keras.optimizers import Adam, SGD, RMSprop
    from nasnet import NASNet
    
    model = NASNet(
        input_shape=(75, 75, 3),
        penultimate_filters=48,
        nb_blocks=2,
        stem_filters=2,
        skip_reduction=True,
        use_auxiliary_branch=False,
        filters_multiplier=1,
        dropout=0.5,
        weight_decay=3e-3,
        include_top=True,
        weights=None,
        input_tensor=None,
        pooling=None,
        classes=2,
        default_size=75)

    
    opt=SGD(lr=0.03, momentum=0.9)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    
    return model


def get_mobile_net():
    from keras.optimizers import Adam, SGD, RMSprop
    from keras.applications.mobilenet import MobileNet
    
    model = MobileNet(
        input_shape=(75, 75, 3),
        alpha=0.5,
        depth_multiplier=1,
        dropout=1e-3,
        include_top=True,
        weights=None,
        input_tensor=None,
        pooling=None,
        classes=2)

    
    opt=SGD(lr=0.03, momentum=0.9)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    
    return model


def prepare_submission(models_proba, path):
    _, test = load_data('../input')
    proba = np.mean(models_proba, axis=0)
    
    submission = pd.DataFrame()
    submission['id'] = test['id']
    submission['is_iceberg'] = proba.reshape((proba.shape[0]))
    submission.to_csv(path, index=False)
    

def main():
    (kfold_data, X_test) = prepare_data('../input')
    
    models_proba = []
    models_roc = []
    models_logloss = []
    
    for idx, data in enumerate(kfold_data):
        X_train, y_train, X_valid, y_valid = data
        
        model = load_model(get_mobile_net)
        callbacks = get_model_callbacks(save_dir=('../experiments/nasnet/fold_%02d' % idx))
        data_generator = get_data_generator(X_train, y_train, batch_size=256)
        
        model.fit_generator(
            data_generator,
            steps_per_epoch=20,
            epochs=5000,
            verbose=True,
            validation_data=(X_valid, y_valid),
            callbacks=callbacks,
            shuffle=True)

        model.load_weights(filepath=('../experiments/nasnet/fold_%02d/model/model_weights.hdf5' % idx))
        proba = model.predict(X_valid)[:, 1]
        
        models_proba.append(model.predict(X_test)[:, 1])
        models_roc.append(roc_auc_score(y_valid.argmax(axis=1), proba))
        models_logloss.append(log_loss(y_valid.argmax(axis=1), proba))
    
    print('ROC AUC:\nMean: %f\nStd: %f\nMin: %f\nMax: %f\n\n' % (np.mean(models_roc), 
                                                             np.std(models_roc),
                                                             np.min(models_roc),
                                                             np.max(models_roc)))

    print('Loss:\nMean: %f\nStd: %f\nMin: %f\nMax: %f\n\n' % (np.mean(models_logloss), 
                                                              np.std(models_logloss),
                                                              np.min(models_logloss),
                                                              np.max(models_logloss)))

    prepare_submission(models_proba, '../submission_nasnet.csv')
    
    
if __name__ == '__main__':
    main()
