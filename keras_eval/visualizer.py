import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.offline import iplot


def plot_confusion_matrix(cm, concepts, normalize=False, fontsize=18, figsize=(16, 12),
                          cmap=plt.cm.coolwarm_r, save_path=None):
    '''

    Args:
        cm: Confusion Matrix, square sized numpy array
        concepts: Name of the categories to show
        normalize: If True, normalize values between 0 and ones. Not valid if negative values.
        fontsize: Text size
        figsize: Figure size
        cmap: Colormap of your choice
        save_path: If `save_path` specified save confusion matrix in that location

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

    # http://stackoverflow.com/questions/21712047/matplotlib-imshow-matshow-display-values-on-plot
    min_val, max_val = 0, len(concepts)
    ind_array = np.arange(min_val, max_val, 1.0)
    x, y = np.meshgrid(ind_array, ind_array)
    for i, (x_val, y_val) in enumerate(zip(x.flatten(), y.flatten())):
        c = cm[int(x_val), int(y_val)]
        ax.text(y_val, x_val, c, va='center', ha='center')

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
