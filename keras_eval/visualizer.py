import os
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import matplotlib.pyplot as plt

from sklearn import metrics
from keras_eval import utils
from plotly.offline import iplot
from sklearn.utils.fixes import signature


def plot_confusion_matrix(cm, concepts, normalize=False, show_text=True, fontsize=18, figsize=(16, 12),
                          cmap=plt.cm.coolwarm_r, save_path=None, show_labels=True):
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
    plt.ylabel('True label', fontweight='bold')
    plt.xlabel('Predicted label', fontweight='bold')

    if show_labels:
        n_labels = len(concepts)
        ax.set_xticklabels(concepts)
        ax.set_yticklabels(concepts)
        plt.xticks(np.arange(0, n_labels, 1.0), rotation='vertical')
        plt.yticks(np.arange(0, n_labels, 1.0))
    else:
        plt.axis('off')

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


def plot_images(image_paths, n_images, title='', subtitles=None, n_cols=5, image_res=(20, 20), save_name=None):
    '''

    Args:
        image_paths: List with image_paths
        n_images: Number of images to show in the plot. Upper bounded by len(image_paths).
        title: Title for the plot
        subtitles: Subtitles for plots
        n_cols: Number of columns to split the data
        image_res: Plot image resolution
        save_name: If specified, will save the plot in save_name path

    Returns: Plots images in the screen

    '''
    if subtitles is not None and len(subtitles) != n_images:
        raise ValueError('Number of images and subtitles is different. There are %d images and %d subtitles'
                         % (n_images, len(subtitles)))
    n_row = 0
    n_col = 0
    total_images_plot = min(len(image_paths), n_images)
    if total_images_plot <= n_cols:
        f, axes = plt.subplots(nrows=1, ncols=n_cols, figsize=image_res)
        plt.title(title)

        for i in range(n_cols):
            if i < total_images_plot:
                img = plt.imread(image_paths[i])
                axes[n_col].imshow(img, aspect='equal')
                axes[n_col].grid('off')
                axes[n_col].axis('off')
                if subtitles is not None:
                    axes[n_col].set_title(subtitles[i])
            else:
                f.delaxes(axes[n_col])
            n_col += 1

    else:
        n_rows_total = int(np.ceil(n_images / n_cols))

        f, axes = plt.subplots(nrows=n_rows_total, ncols=n_cols, figsize=image_res)

        for i in range(n_rows_total * n_cols):
            if i < total_images_plot:
                img = plt.imread(image_paths[i])
                axes[n_row, n_col].imshow(img, aspect='equal')
                axes[n_row, n_col].grid('off')
                axes[n_row, n_col].axis('off')
                if subtitles is not None:
                    axes[n_row, n_col].set_title(subtitles[i])
            else:
                axes[n_row, n_col].grid('off')
                f.delaxes(axes[n_row, n_col])

            n_col += 1

            if n_col == n_cols:
                n_col = 0
                n_row += 1

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


def plot_models_performance(eval_dir, individual=False, class_idx=None, metric=None, save_name=None):
    '''
       Enables plotting of a single metric from multiple evaluation metrics files
       Args:
           eval_dir: A directory that contains multiple metrics files
           individual: If True, compare individual metrics. Otherwise, compare average metrics.
           class_idx: The index of class for when comparing individual metrics
           metric: The metric to be plotted for comparison
           save_name:  If `save_path` specified, save plot in that location

       Returns: Nothing. If save_path is provided, plot is stored.

    '''
    x_axis = []
    y_axis = []
    tick_label = []
    i = 0
    for result_csv in os.listdir(eval_dir):
        if utils.check_result_type(result_csv, individual):
            df = pd.read_csv(os.path.join(eval_dir, result_csv))
            tick_label.append(result_csv[:result_csv.rfind('_')])
            if individual:
                if isinstance(class_idx, int) and isinstance(metric, str):
                    y_axis.append(df[metric][class_idx])
                    x_axis.append(i)
                else:
                    raise ValueError('Unsupported type: class_idx, metric')
            else:
                if metric:
                    y_axis.append(df[metric][0])
                    x_axis.append(i)
                else:
                    raise ValueError('Missing required option: metric')
            i += 1
    plt.bar(x_axis, y_axis)
    plt.ylabel(str(metric))
    plt.xticks(x_axis, tick_label, rotation='vertical')
    if save_name:
        plt.savefig(save_name)


def plot_confidence_interval(values_x, values_y, lower_bound, upper_bound, title=''):
    if len(values_x) == len(values_y) == len(lower_bound) == len(upper_bound):
        upper_bound = go.Scatter(
            name='Upper Bound',
            x=values_x,
            y=upper_bound,
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=0),
            fillcolor='rgba(25, 25, 255, 0.2)',
            fill='tonexty')

        trace = go.Scatter(
            name='Mean',
            x=values_x,
            y=values_y,
            mode='lines',
            line=dict(color='rgb(31, 119, 180)'),
            fillcolor='rgba(25, 25, 255, 0.2)',
            fill='tonexty')

        lower_bound = go.Scatter(
            name='Lower Bound',
            x=values_x,
            y=lower_bound,
            marker=dict(color="#444"),
            line=dict(width=0),
            mode='lines')

        data = [lower_bound, trace, upper_bound]

        layout = go.Layout(
            xaxis=dict(title='Top-1 Probability'),
            yaxis=dict(title='Confidence Interval'),
            title=title,
            showlegend=False)

        fig = go.Figure(data=data, layout=layout)
        iplot(fig, filename='confidence_interval')
    else:
        raise ValueError('Arrays "values_x", "values_y", "lower_bound" and '
                         '"upper_bound" should have the same dimension')


def plot_histogram(data, bins, title, xlabel, ylabel):
    plt.hist(data, bins=bins)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)


def scatter_plot(values_x, values_y, axis_x, axis_y, title):
    if len(values_x) != len(values_y):
        raise ValueError('Both arrays "values_x" and "values_y" should have the same dimension')

    data = [go.Scatter(
        x=values_x,
        y=values_y,
        mode='markers',
        marker=dict(
            color='rgba(156, 165, 196, 0.95)',
            line=dict(
                color='rgba(156, 165, 196, 1.0)',
                width=1,
            ),
            symbol='circle',
            size=8,
        )
    )]

    layout = dict(title=title,
                  xaxis=dict(title=axis_x),
                  yaxis=dict(title=axis_y),
                  )

    fig = dict(data=data, layout=layout)
    iplot(fig, filename='scatter-plot')
