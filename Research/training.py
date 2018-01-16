import os

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, average_precision_score
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau

from data_loader import prepare_data, prepare_data_cv, load_data
from data_generation import get_data_generator
from plots import plot_precision_recall, plot_roc, plot_confusion_matrix


def logloss_softmax(y_true, y_pred, eps=1e-15):
    proba = y_pred[np.arange(len(y_pred)), np.argmax(y_true, axis=1)]
    proba = np.clip(proba, eps, 1 - eps)
    return -np.mean(np.log(proba))


def get_model_callbacks(save_dir):
    stopping = EarlyStopping(monitor='val_loss', 
                             min_delta=1e-2,
                             patience=30,
                             verbose=False, 
                             mode='min')
    
    board_path = os.path.join(save_dir, 'board')
    if not os.path.exists(board_path):
        os.makedirs(board_path)
    
    board = TensorBoard(log_dir=board_path)
    
    lr_sheduler = ReduceLROnPlateau(monitor='val_loss',
                                    factor=0.3,
                                    patience=10,
                                    verbose=True,
                                    mode='min', 
                                    epsilon=1e-2,
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
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    model.summary()
    return model


def get_resnext():
    from resnext import ResNext

    model= ResNext(
        input_shape=(75, 75, 3),
        depth=11,
        cardinality=4,
        width=4,
        weight_decay=0.,
        include_top=True, 
        weights=None,
        classes=2)
    
    return model


def get_resnet_18():
    from resnet import ResnetBuilder

    model = ResnetBuilder.build_resnet_50(input_shape=(3, 75, 75),
                                          num_outputs=2,
                                          filters=16,
                                          weight_decay=1e-4)

    return model


def get_class_weights(y):
    true_labels_count = np.sum(y[:, 1])
    false_labels_count = y.shape[0] - true_labels_count
    scaler = max(true_labels_count, false_labels_count)

    return {
        1:  scaler / false_labels_count,
        0:  scaler / true_labels_count,
    }


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


def main(experiment_path, plot_results=False):
    (kfold_data, X_test) = prepare_data_cv('../input')
    
    models_proba = []
    models_roc = []
    models_logloss = []
    models_map = []
    
    for idx, data in enumerate(kfold_data):
        X_train, y_train, X_valid, y_valid = data
        
        model = load_model(get_resnet_18, weights=None)
        callbacks = get_model_callbacks(save_dir=os.path.join(experiment_path, 'fold_%02d' % idx))
        data_generator = get_data_generator(X_train, y_train, batch_size=128)
        # class_weights = get_class_weights(y_train)
        # print(class_weights)

        model.fit_generator(
            data_generator,
            # class_weight=class_weights,
            steps_per_epoch=10,
            epochs=2000,
            verbose=True,
            validation_data=(X_valid, y_valid),
            callbacks=callbacks,
            shuffle=True)

        model.load_weights(filepath=os.path.join(experiment_path, ('fold_%02d/model/model_weights.hdf5' % idx)))
        proba = model.predict(X_valid)
        proba_test = model.predict(X_test)[:, 1]
        
        models_proba.append(proba_test)
        models_roc.append(roc_auc_score(y_valid.argmax(axis=1), proba[:, 1]))
        models_map.append(average_precision_score(y_valid.argmax(axis=1), proba[:, 1]))
        models_logloss.append(logloss_softmax(y_valid, proba))

        prepare_submission([proba_test], os.path.join(experiment_path, 'fold_%02d/prediction.csv' % idx))

        if plot_results:
            plots_path = os.path.join(experiment_path, 'fold_%02d/plots' % idx)
            if not os.path.exists(plots_path):
                os.makedirs(plots_path)

            plot_precision_recall(proba[:, 1], y_valid.argmax(axis=1),
                                  path=os.path.join(plots_path, 'recall_precision.jpg'))

            plot_roc(proba[:, 1], y_valid.argmax(axis=1),
                     path=os.path.join(plots_path, 'roc.jpg'))

            plot_confusion_matrix(proba[:, 1], y_valid.argmax(axis=1),
                                  path=os.path.join(plots_path, 'conf.jpg'))

        print('Loss:\nMean: %f\nStd: %f\nMin: %f\nMax: %f\n\n' % (np.mean(models_logloss),
                                                                  np.std(models_logloss),
                                                                  np.min(models_logloss),
                                                                  np.max(models_logloss)))

        print('ROC AUC:\nMean: %f\nStd: %f\nMin: %f\nMax: %f\n\n' % (np.mean(models_roc),
                                                                     np.std(models_roc),
                                                                     np.min(models_roc),
                                                                     np.max(models_roc)))

        print('mAP:\nMean: %f\nStd: %f\nMin: %f\nMax: %f\n\n' % (np.mean(models_map),
                                                                 np.std(models_map),
                                                                 np.min(models_map),
                                                                 np.max(models_map)))

        raise NotImplementedError(":3")

    prepare_submission(models_proba, os.path.join(experiment_path, 'submission.csv'))
    
    
if __name__ == '__main__':
    main(experiment_path='../experiments/resnet_50_new', plot_results=True)
