import itertools

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix


def plot_precision_recall(y_pred, y_true, path):
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


def _plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_confusion_matrix(y_predicted, y_true, path):
    # Compute confusion matrix

    y_predicted = (y_predicted > 0.5).astype(np.int32)
    cnf_matrix = confusion_matrix(y_true, y_predicted)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    _plot_confusion_matrix(cnf_matrix, classes={0: 0, 1: 1},
                           title='Confusion matrix, without normalization')

    plt.savefig(path, dpi=80)
