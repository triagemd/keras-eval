import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(probs, labels, class_names, fontsize=18, figsize=(16, 12), cmap=plt.cm.coolwarm_r, save_path=None):
    '''

    Args:
       probs: Output of the CNN
       labels: Ground truth classes (categorical)
       class_names: List of strings containing classes names
       fontsize: Size of text
       figsize: Size of figure
       cmap: Color choice
       save_path: If `save_path` specified save confusion matrix in that location

    Returns: Nothing, shows confusion matrix

    '''

    y_pred = np.argmax(probs, axis=1)
    y_true = np.argmax(labels, axis=1)

    n_labels = len(class_names)
    np.set_printoptions(precision=2)
    plt.rcParams.update({'font.size': fontsize})

    cm = confusion_matrix(y_true, y_pred, range(len(class_names)))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm_normalized, vmin=0, vmax=1, alpha=0.8, cmap=cmap)

    fig.colorbar(cax)
    ax.xaxis.tick_bottom()
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    plt.xticks(np.arange(0, n_labels, 1.0), rotation='vertical')
    plt.yticks(np.arange(0, n_labels, 1.0))
    plt.ylabel('True label', fontweight='bold')
    plt.xlabel('Predicted label', fontweight='bold')

    # http://stackoverflow.com/questions/21712047/matplotlib-imshow-matshow-display-values-on-plot
    min_val, max_val = 0, len(class_names)
    ind_array = np.arange(min_val, max_val, 1.0)
    x, y = np.meshgrid(ind_array, ind_array)
    for i, (x_val, y_val) in enumerate(zip(x.flatten(), y.flatten())):
        c = cm[int(x_val), int(y_val)]
        ax.text(y_val, x_val, c, va='center', ha='center')

    if save_path is not None:
        plt.savefig(save_path)


def plot_threshold_impact(correct_array, errors_array, threshold):
    '''

    Args:
        correct_array: Array of correct predictions per each threshold
        errors_array: Array of error predictions per each threshold
        threshold: Threshold values used

    Returns:

    '''

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
