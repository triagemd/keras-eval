import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import plotly.graph_objs as go
from plotly.offline import iplot


def plot_confusion_matrix(probabilities, labels, concept_labels, fontsize=18, figsize=(16, 12), cmap=plt.cm.coolwarm_r, save_path=None):
    '''

    Args:
       probabilities: Output of the CNN
       labels: Ground truth classes (categorical)
       concept_labels: List of strings containing classes names
       fontsize: Size of text
       figsize: Size of figure
       cmap: Color choice
       save_path: If `save_path` specified save confusion matrix in that location

    Returns: Nothing, shows confusion matrix

    '''

    y_pred = np.argmax(probabilities, axis=1)
    y_true = np.argmax(labels, axis=1)

    n_labels = len(concept_labels)
    np.set_printoptions(precision=2)
    plt.rcParams.update({'font.size': fontsize})

    cm = confusion_matrix(y_true, y_pred, range(len(concept_labels)))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm_normalized, vmin=0, vmax=1, alpha=0.8, cmap=cmap)

    fig.colorbar(cax)
    ax.xaxis.tick_bottom()
    ax.set_xticklabels(concept_labels)
    ax.set_yticklabels(concept_labels)
    plt.xticks(np.arange(0, n_labels, 1.0), rotation='vertical')
    plt.yticks(np.arange(0, n_labels, 1.0))
    plt.ylabel('True label', fontweight='bold')
    plt.xlabel('Predicted label', fontweight='bold')

    # http://stackoverflow.com/questions/21712047/matplotlib-imshow-matshow-display-values-on-plot
    min_val, max_val = 0, len(concept_labels)
    ind_array = np.arange(min_val, max_val, 1.0)
    x, y = np.meshgrid(ind_array, ind_array)
    for i, (x_val, y_val) in enumerate(zip(x.flatten(), y.flatten())):
        c = cm[int(x_val), int(y_val)]
        ax.text(y_val, x_val, c, va='center', ha='center')

    if save_path is not None:
        plt.savefig(save_path)


def plotly_threshold(threshold, correct, errors, title='Threshold Tuning'):
    '''

    Args:
        threshold: List of thresholds
        correct: List of correct predictions per threshold
        errors: List of error predictions per threshold
        title: Title of the plot

    Returns: Interactive Plot

    '''
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


def plot_images(image_paths, n_imgs, title='', save_name=None):
    n_row = 0
    n_col = 0

    if n_imgs <= 5:
        f, axes = plt.subplots(nrows=1, ncols=n_imgs, figsize=(n_imgs, n_imgs))
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

        f, axes = plt.subplots(nrows=n_rows_total, ncols=5, figsize=(n_imgs, n_imgs))
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
