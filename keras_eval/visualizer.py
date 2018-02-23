import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow.contrib.tensorboard.plugins import projector


def confusion_matrix(y_pred, y_true, labels, fontsize=18, figsize=(16, 12), cmap=plt.cm.coolwarm_r, save_path=None):
    '''

    Args:
        y_true:
        y_pred:
        labels:
        fontsize:
        figsize:
        cmap:
        save_path:

    Returns:

    '''

    n_labels = len(labels)
    np.set_printoptions(precision=2)
    plt.rcParams.update({'font.size': fontsize})

    cm = confusion_matrix(y_true, y_pred, range(len(labels)))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm_normalized, vmin=0, vmax=1, alpha=0.8, cmap=cmap)

    fig.colorbar(cax)
    ax.xaxis.tick_bottom()
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    plt.xticks(np.arange(0, n_labels, 1.0), rotation='vertical')
    plt.yticks(np.arange(0, n_labels, 1.0))
    plt.ylabel('True label', fontweight='bold')
    plt.xlabel('Predicted label', fontweight='bold')

    # http://stackoverflow.com/questions/21712047/matplotlib-imshow-matshow-display-values-on-plot
    min_val, max_val = 0, len(labels)
    ind_array = np.arange(min_val, max_val, 1.0)
    x, y = np.meshgrid(ind_array, ind_array)
    for i, (x_val, y_val) in enumerate(zip(x.flatten(), y.flatten())):
        c = cm[int(x_val), int(y_val)]
        ax.text(y_val, x_val, c, va='center', ha='center')

    if save_path is not None:
        plt.savefig(save_path)


def threshold_impact(correct_array, errors_array, threshold):

    n_errors = errors_array[0]
    n_correct = correct_array[0]

    threshold = np.array(threshold)

    errors_percentage = ((n_errors - errors_array) / n_errors) * 100
    corrects_percentage = (correct_array / n_correct) * 100

    plt.plot(threshold, errors_percentage, color='b', label='Removed Errors')
    plt.plot(threshold, corrects_percentage, color='r', label='Correct Predictions')
    plt.xlabel('Threshold values')
    plt.ylabel('%')
    plt.legend()
