import os

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, average_precision_score
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau

from data_loader import prepare_data, prepare_data_cv, load_data
from data_generation import get_data_generator


def logloss_softmax(y_true, y_pred, eps=1e-15):
    proba = y_pred[np.arange(len(y_pred)), np.argmax(y_true, axis=1)]
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

    model = ResnetBuilder.build_resnet_18(input_shape=(3, 75, 75),
                                          num_outputs=2,
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


def plot_precision_recall(y_pred, y_true, path):
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve

    precision, recall, _ = precision_recall_curve(y_true, y_pred)

    plt.figure(figsize=(10, 10))
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.grid()
    plt.title('2-class Precision-Recall curve: AP={0:0.6f}'.format(average_precision_score(y_true, y_pred)))
    plt.savefig(path, dpi=80)


def plot_roc(y_pred, y_true, path):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_pred)

    plt.figure(figsize=(10, 10))
    plt.step(fpr, tpr, color='b', alpha=0.2,
             where='post')
    plt.plot([0., 1.], [0., 1.], color='navy', linestyle='--')

    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.grid()
    plt.title('ROC AUC = {0:0.6f}'.format(roc_auc_score(y_true, y_pred)))
    plt.savefig(path, dpi=80)


def main(experiment_path, plot_results=False):
    (kfold_data, X_test) = prepare_data_cv('../input')
    
    models_proba = []
    models_roc = []
    models_logloss = []
    models_map = []
    
    for idx, data in enumerate(kfold_data):
        X_train, y_train, X_valid, y_valid = data
        
        model = load_model(get_resnext, weights='../experiments/resnext_02/fold_%02d/model/model_weights.hdf5' % idx)
        callbacks = get_model_callbacks(save_dir=os.path.join(experiment_path, 'fold_%02d' % idx))
        data_generator = get_data_generator(X_train, y_train, batch_size=64)

        model.fit_generator(
            data_generator,
            steps_per_epoch=30,
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

    prepare_submission(models_proba, os.path.join(experiment_path, 'submission.csv'))
    
    
if __name__ == '__main__':
    main(experiment_path='../experiments/resnext_02', plot_results=True)
