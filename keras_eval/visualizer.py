import numpy as np
import plotly.graph_objs as go
import matplotlib.pyplot as plt

from sklearn import metrics
from keras_eval import utils
from plotly.offline import iplot
from sklearn.utils.fixes import signature


def plot_confusion_matrix(cm, concepts, normalize=False, show_text=True, fontsize=18, figsize=(16, 12),
                          cmap=plt.cm.coolwarm_r, save_path=None):
    '''
    Plot confusion matrix provided in 'cm'

    Args:
        cm: Confusion Matrix, square sized numpy array
        concepts: Name of the categories to show
        normalize: If True, normalize values between 0 and ones. Not valid if negative values.
        show_text: If True, display cell values as text. Otherwise only display cell colors.
        fontsize: Size of text
        figsize: Size of figure
        cmap: Color choice
        save_path: If `save_path` specified, save confusion matrix in that location

    Returns: Nothing. Plots confusion matrix

    '''

    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError('Invalid confusion matrix shape, it should be square and ndim=2')

    if cm.shape[0] != len(concepts) or cm.shape[1] != len(concepts):
        raise ValueError('Number of concepts (%i) and dimensions of confusion matrix do not coincide (%i, %i)' %
                         (len(concepts), cm.shape[0], cm.shape[1]))

    plt.rcParams.update({'font.size': fontsize})

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    if normalize:
        cm = cm_normalized

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm, vmin=np.min(cm), vmax=np.max(cm), alpha=0.8, cmap=cmap)

    fig.colorbar(cax)
    ax.xaxis.tick_bottom()
    n_labels = len(concepts)
    ax.set_xticklabels(concepts)
    ax.set_yticklabels(concepts)
    plt.xticks(np.arange(0, n_labels, 1.0), rotation='vertical')
    plt.yticks(np.arange(0, n_labels, 1.0))
    plt.ylabel('True label', fontweight='bold')
    plt.xlabel('Predicted label', fontweight='bold')

    if show_text:
        # http://stackoverflow.com/questions/21712047/matplotlib-imshow-matshow-display-values-on-plot
        min_val, max_val = 0, len(concepts)
        ind_array = np.arange(min_val, max_val, 1.0)
        x, y = np.meshgrid(ind_array, ind_array)
        for i, (x_val, y_val) in enumerate(zip(x.flatten(), y.flatten())):
            c = cm[int(x_val), int(y_val)]
            ax.text(y_val, x_val, c, va='center', ha='center')

    if save_path is not None:
        plt.savefig(save_path)


def plot_ROC_curve(y_probs, y_true, title='ROC curve', save_path=None):
    """
    Plot Receiver Operating Characteristic (ROC) curve understood as true positive rate (TPR) against the
    false positive rate (FPR) at various threshold settings.

    Note: this implementation is restricted to the binary classification task.

    Args:
        y_probs: A numpy array containing the probabilities of the positive class.
        y_true: A numpy array of the true binary labels (*not* encoded as 1-hot).
        title: String with the title.
        save_path: If `save_path` specified, save confusion matrix in that location

    Returns: Nothing, displays ROC curve
    """
    utils.check_input_samples(y_probs, y_true)

    if not np.array_equal(y_true, y_true.astype(bool)):
        raise ValueError('y_true must contain the true binary labels.')

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_probs)
    roc_auc = metrics.auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title(title)
    plt.legend(loc="lower right")

    if save_path is not None:
        plt.savefig(save_path)


def plot_precision_recall_curve(y_probs, y_true, title='2-class Precision-Recall curve', save_path=None):
    """
    Plot Precision-Recall curve for a binary classification task.

    Note: this implementation is restricted to the binary classification task.

    Args:
        y_probs: A numpy array containing the probabilities of the positive class.
        y_true: A numpy array of the true binary labels (*not* encoded as 1-hot).
        title: String with the title.
        save_path: If `save_path` specified, save confusion matrix in that location.

    Returns: Nothing, displays Precision-Recall curve
    """
    if not np.array_equal(y_true, y_true.astype(bool)):
        raise ValueError('y_true must contain the true binary labels.')

    precision, recall, _ = metrics.precision_recall_curve(y_true, y_probs)
    average_precision = metrics.average_precision_score(y_true, y_probs)

    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title(title + ': AP={0:0.2f}'.format(average_precision))

    if save_path is not None:
        plt.savefig(save_path)


def plot_threshold(threshold, correct, errors, title='Threshold Tuning'):
    '''

    Args:
        threshold: List of thresholds
        correct: List of correct predictions per threshold
        errors: List of error predictions per threshold
        title: Title of the plot

    Returns: Interactive Plot

    '''
    if not len(threshold) == len(correct) == len(errors):
        raise ValueError('The length of the arrays introduced do not coincide (%i), (%i), (%i)'
                         % (len(threshold), len(correct), len(errors)))

    trace1 = go.Scatter(
        x=threshold,
        y=correct,
        name='Correct Predictions'
    )

    trace2 = go.Scatter(
        x=threshold,
        y=errors,
        name='Removed Errors'
    )

    layout = dict(title=title,
                  xaxis=dict(title='Threshold Value'),
                  yaxis=dict(title='Network Predictions (%)'),
                  )

    data = [trace1, trace2]
    fig = dict(data=data, layout=layout)
    iplot(fig, filename='Threshold Tuning')


def plot_images(image_paths, n_imgs, title='', image_res=(20, 20), save_name=None):
    n_row = 0
    n_col = 0

    if n_imgs <= 5:
        f, axes = plt.subplots(nrows=1, ncols=n_imgs, figsize=image_res)
        plt.title(title)
        for i, image_path in enumerate(image_paths):

            if i == n_imgs:
                break

            img = plt.imread(image_path)
            axes[n_col].imshow(img, aspect='equal')
            axes[n_col].grid('off')
            axes[n_col].axis('off')
            n_col += 1

    else:
        n_rows_total = int(np.ceil(n_imgs / 5))

        f, axes = plt.subplots(nrows=n_rows_total, ncols=5, figsize=image_res)
        plt.title(title)
        for i, image_path in enumerate(image_paths):

            if i == n_imgs:
                break

            img = plt.imread(image_path)
            axes[n_row, n_col].imshow(img, aspect='equal')
            axes[n_row, n_col].grid('off')
            axes[n_row, n_col].axis('off')
            n_row += 1
            if n_row == int(np.ceil(n_imgs / 5)):
                n_row = 0
                n_col += 1

    if save_name is not None:
        plt.savefig(save_name)


def plot_concept_metrics(concepts, metrics, x_axis_label, y_axis_label, title=None):
    if len(concepts) != len(metrics):
        raise ValueError('Dimensions of concepts (%i) and metrics array (%i) do not match' % (len(concepts),
                                                                                              len(metrics)))
    data = [[] for i in range(len(concepts))]
    for i in range(len(concepts)):
        data[i] = go.Scatter(
            x=np.arange(1, len(metrics[i]) + 1),
            y=metrics[i],
            mode='lines',
            name=concepts[i],
        )
    layout = go.Layout(
        title=title,
        xaxis=dict(
            title=x_axis_label
        ),
        yaxis=dict(
            title=y_axis_label
        ),
    )
    fig = go.Figure(data=data, layout=layout)
    iplot(fig, filename='line-mode')
