import os

import numpy as np
import pandas as pd

from sklearn.metrics import log_loss, roc_auc_score

from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau

from data_loader import prepare_data, prepare_data_cv, load_data
from base_model import get_base_model
from data_generation import get_data_generator


def logloss_softmax(y_true, y_pred):
    proba = y_pred[:, np.argmax(y_true, axis=1)]
    proba = np.maximum(np.minimum(1 - 1e-15, proba), 1e-15)
    return -np.average(np.log(proba))


def logloss_sigmoid(y_true, y_pred):
    proba = np.maximum(np.minimum(1 - 1e-15, y_pred), 1e-15)
    return -np.average(np.log(proba) * y_true + (1 - y_true) * np.log(1 - proba))
    

def get_model_callbacks(save_dir):
    stopping = EarlyStopping(monitor='val_loss', 
                             min_delta=1e-3, 
                             patience=60, 
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
    from keras.optimizers import Adam, SGD, RMSprop
    from resnext import ResNext
    model= ResNext(
        input_shape=(75, 75, 3),
        depth=29,
        cardinality=4, 
        width=4,
        weight_decay=1e-2,
        include_top=True, 
        weights=None,
        classes=1)
    
    opt=SGD(lr=0.01, momentum=0.9)
    model.compile(loss='mae',
                  optimizer=opt,
                  metrics=['accuracy'])
    
    return model


def get_nasnet():
    from keras.optimizers import Adam, SGD, RMSprop
    from nasnet import NASNet
    
    model = NASNet(
        input_shape=(75, 75, 3),
        penultimate_filters=24,
        nb_blocks=1,
        stem_filters=2,
        skip_reduction=True,
        use_auxiliary_branch=False,
        filters_multiplier=2,
        dropout=0.0,
        weight_decay=1e-2,
        include_top=True,
        weights=None,
        input_tensor=None,
        pooling=None,
        classes=2,
        default_size=75)

    
    opt=SGD(lr=0.01, momentum=0.9)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model


def get_resnet_18():
    from resnet import ResnetBuilder
    from keras.optimizers import Adam, SGD, RMSprop
    
    model = ResnetBuilder.build_resnet_18(input_shape=(3, 75, 75),
                                          num_outputs=1,
                                          weight_decay=5e-4)
    
    opt=Adam(lr=0.001)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model


def get_resnet_34():
    from resnet import ResnetBuilder
    from keras.optimizers import Adam, SGD, RMSprop
    
    model = ResnetBuilder.build_resnet_34(input_shape=(3, 75, 75),
                                          num_outputs=1,
                                          weight_decay=5e-3)
    
    opt=SGD(lr=0.001, momentum=0.9)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model


def get_dense_net():
    from densenet import DenseNet
    from keras.optimizers import Adam, SGD, RMSprop
    
    model = DenseNet(
        nb_dense_block=4, 
        growth_rate=32, 
        nb_filter=64, 
        reduction=0.0, 
        dropout_rate=0.0,
        weight_decay=1e-4, 
        classes=1, 
        weights_path=None)
    
    opt=SGD(lr=0.03, momentum=0.9)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model


def get_shallow():
    from shallow import get_shallow_model
    from keras.optimizers import Adam, SGD, RMSprop
    
    model = get_shallow_model(weight_decay=0.)
    
    opt=SGD(lr=0.001, momentum=0.9)
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
    (kfold_data, X_test) = prepare_data_cv('../input')
    
    models_proba = []
    models_roc = []
    models_logloss = []
    
    for idx, data in enumerate(kfold_data):
        X_train, y_train, X_valid, y_valid = data
        
        model = load_model(get_resnet_18)
        callbacks = get_model_callbacks(save_dir=('../experiments/resnet_18_cv/fold_%02d' % idx))
        data_generator = get_data_generator(X_train, y_train, batch_size=64)
        
        model.fit_generator(
            data_generator,
            steps_per_epoch=40,
            epochs=2000,
            verbose=True,
            validation_data=(X_valid, y_valid),
            callbacks=callbacks,
            shuffle=True)

        model.load_weights(filepath=('../experiments/resnet_18_cv/fold_%02d/model/model_weights.hdf5' % idx))
        proba = model.predict(X_valid)
        
        models_proba.append(model.predict(X_test))
        models_roc.append(roc_auc_score(y_valid, proba))
        models_logloss.append(logloss_sigmoid(y_valid, proba))
        
        prepare_submission(models_proba, ('../submission_resnet_18_%2d.csv' % idx))
    
    print('ROC AUC:\nMean: %f\nStd: %f\nMin: %f\nMax: %f\n\n' % (np.mean(models_roc), 
                                                                 np.std(models_roc),
                                                                 np.min(models_roc),
                                                                 np.max(models_roc)))

    print('Loss:\nMean: %f\nStd: %f\nMin: %f\nMax: %f\n\n' % (np.mean(models_logloss), 
                                                              np.std(models_logloss),
                                                              np.min(models_logloss),
                                                              np.max(models_logloss)))
    
if __name__ == '__main__':
    main()
