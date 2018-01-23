import os

import numpy as np
import pandas as pd

from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from data_loader import load_data


def logloss_softmax(y_true, y_pred, eps=1e-15):
    proba = y_pred[np.arange(len(y_pred)), np.argmax(y_true, axis=1)]
    proba = np.clip(proba, eps, 1 - eps)
    return -np.mean(np.log(proba))


def get_model_callbacks(save_dir):
    stopping = EarlyStopping(monitor='val_loss',
                             min_delta=1e-3,
                             patience=45,
                             verbose=False,
                             mode='min')

    board_path = os.path.join(save_dir, 'board')
    if not os.path.exists(board_path):
        os.makedirs(board_path)

    board = TensorBoard(log_dir=board_path)

    lr_sheduler = ReduceLROnPlateau(monitor='val_loss',
                                    factor=0.3,
                                    patience=15,
                                    verbose=True,
                                    mode='min',
                                    epsilon=5e-3,
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


def load_model(model_loader_fn, weights=None):
    from keras.optimizers import RMSprop

    model = model_loader_fn()

    if weights:
        model.load_weights(weights)

    opt = RMSprop(lr=1e-3)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    model.summary()
    return model


def get_resnext():
    from resnext import ResNext

    model = ResNext(
        input_shape=(75, 75, 3),
        depth=29,
        cardinality=8,
        width=4,
        weight_decay=0.,
        include_top=True,
        weights=None,
        classes=2)

    return model


def get_resnet_18():
    from resnet import ResnetBuilder

    model = ResnetBuilder.build_resnet_18(input_shape=(3, 75, 75),
                                          num_outputs=2,
                                          filters=8,
                                          weight_decay=0.)

    return model


def prepare_submission(models_proba, path, high_thr=0.9, low_thr=0.1):
    _, test = load_data('../input')
    models_proba = np.array(models_proba)
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
