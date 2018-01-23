import os

import numpy as np

from sklearn.metrics import roc_auc_score, average_precision_score

from data_loader import prepare_data, prepare_data_cv
from data_generation import get_data_generator
from plots import plot_precision_recall, plot_roc, plot_confusion_matrix
from utils import logloss_softmax, get_model_callbacks, load_model
from utils import get_resnet_18, get_resnext, prepare_submission


def main(experiment_path, plot_results=False):
    (kfold_data, X_test) = prepare_data_cv('../input')
    
    models_proba = []
    models_acc = []
    models_roc = []
    models_logloss = []
    models_map = []
    
    for idx, data in enumerate(kfold_data):
        X_train, y_train, X_valid, y_valid = data
        
        model = load_model(get_resnet_18, weights=None)
        callbacks = get_model_callbacks(save_dir=os.path.join(experiment_path, 'fold_%02d' % idx))
        data_generator = get_data_generator(X_train, y_train, batch_size=128)

        model.fit_generator(
            data_generator,
            steps_per_epoch=10,
            epochs=1000,
            verbose=True,
            validation_data=(X_valid, y_valid),
            callbacks=callbacks,
            shuffle=True)

        model.load_weights(filepath=os.path.join(experiment_path, ('fold_%02d/model/model_weights.hdf5' % idx)))

        _, acc_val = model.evaluate(X_valid, y_valid, verbose=False)
        proba = model.predict(X_valid)
        proba_test = model.predict(X_test)[:, 1]

        models_proba.append(proba_test)
        models_acc.append(acc_val)
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

        print('Acc:\nMean: %f\nStd: %f\nMin: %f\nMax: %f\n\n' % (np.mean(models_acc),
                                                                  np.std(models_acc),
                                                                  np.min(models_acc),
                                                                  np.max(models_acc)))

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
    main(experiment_path='../experiments/resnext_038', plot_results=False)
