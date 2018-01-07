import os

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score

from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau

from data_loader import prepare_data, prepare_data_cv, load_data
from data_generation import get_data_generator


def logloss_softmax(y_true, y_pred, eps=1e-15):
    proba = y_pred[:, np.argmax(y_true, axis=1)]
    proba = np.clip(proba, eps, 1 - eps)
    return -np.mean(np.log(proba))


def get_model_callbacks(save_dir):
    stopping = EarlyStopping(monitor='val_loss', 
                             min_delta=1e-3,
                             patience=50,
                             verbose=False, 
                             mode='min')
    
    board_path = os.path.join(save_dir, 'board')
    if not os.path.exists(board_path):
        os.makedirs(board_path)
    
    board = TensorBoard(log_dir=board_path)
    
    lr_sheduler = ReduceLROnPlateau(monitor='val_loss', 
                                    factor=0.5,
                                    patience=20,
                                    verbose=True,
                                    mode='min', 
                                    epsilon=1e-3,
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
    from keras.optimizers import RMSprop, Adam
    from resnext import ResNext

    model= ResNext(
        input_shape=(75, 75, 3),
        depth=20,
        cardinality=3,
        width=5,
        weight_decay=0.,
        include_top=True, 
        weights=None,
        classes=2)
    
    opt=Adam(lr=1e-2)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    
    return model


def get_resnet_18():
    from resnet import ResnetBuilder
    from keras.optimizers import RMSprop, Adam

    model = ResnetBuilder.build_resnet_18(input_shape=(3, 75, 75),
                                          num_outputs=2,
                                          weight_decay=0.)

    opt = RMSprop(lr=2e-3)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model


def prepare_submission(models_proba, path, high_thr=0.9, low_thr=0.1):
    _, test = load_data('../input')
    proba = np.where(np.all(models_proba > high_thr, axis=0),
                     np.max(models_proba, axis=0),
                     np.where(np.all(models_proba < low_thr, axis=0),
                              np.min(models_proba, axis=0),
                              np.median(models_proba, axis=0))
                    )
    
    submission = pd.DataFrame()
    submission['id'] = test['id']
    submission['is_iceberg'] = proba.reshape((proba.shape[0]))
    submission.to_csv(path, index=False)
    

def main():
    (kfold_data, X_test) = prepare_data_cv('../input')
    
    models_proba = []
    models_roc = []
    models_logloss = []
    
    for idx, data in enumerate(kfold_data):
        X_train, y_train, X_valid, y_valid = data
        
        model = load_model(get_resnext)
        callbacks = get_model_callbacks(save_dir=('../experiments/resnext_04/fold_%02d' % idx))
        data_generator = get_data_generator(X_train, y_train, batch_size=64)

        model.fit_generator(
            data_generator,
            steps_per_epoch=30,
            epochs=2000,
            verbose=True,
            validation_data=(X_valid, y_valid),
            callbacks=callbacks,
            shuffle=True)

        model.load_weights(filepath=('../experiments/resnext_04/fold_%02d/model/model_weights.hdf5' % idx))
        proba = model.predict(X_valid)
        proba_test = model.predict(X_test)[:, 1]
        
        models_proba.append(proba_test)
        models_roc.append(roc_auc_score(y_valid.argmax(axis=1), proba[:, 1]))
        models_logloss.append(logloss_softmax(y_valid, proba))

        prepare_submission([proba_test], ('../resnext_04_%02d.csv') % idx)
    
    print('ROC AUC:\nMean: %f\nStd: %f\nMin: %f\nMax: %f\n\n' % (np.mean(models_roc), 
                                                                 np.std(models_roc),
                                                                 np.min(models_roc),
                                                                 np.max(models_roc)))

    print('Loss:\nMean: %f\nStd: %f\nMin: %f\nMax: %f\n\n' % (np.mean(models_logloss), 
                                                              np.std(models_logloss),
                                                              np.min(models_logloss),
                                                              np.max(models_logloss)))

    prepare_submission(models_proba, '../resnext_04.csv')
    
    
if __name__ == '__main__':
    main()
