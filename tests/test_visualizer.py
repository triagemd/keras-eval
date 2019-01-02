import pytest
import numpy as np

from keras_eval.visualizer import plot_confusion_matrix, plot_ROC_curve, plot_precision_recall_curve,\
    plot_concept_metrics, plot_threshold, plot_models_performance, scatter_plot, plot_confidence_interval


def test_plot_confusion_matrix():
    confusion_matrix = np.ones((6, 5))
    concepts = ['a', 'b', 'c', 'd', 'e']
    with pytest.raises(ValueError) as exception:
        plot_confusion_matrix(confusion_matrix, concepts)
    expected = 'Invalid confusion matrix shape, it should be square and ndim=2'
    actual = str(exception).split('ValueError: ')[1]
    assert actual == expected

    confusion_matrix = np.ones((5, 5))
    concepts = ['a', 'b', 'c', 'd', 'e', 'f']
    with pytest.raises(ValueError) as exception:
        plot_confusion_matrix(confusion_matrix, concepts)
    expected = 'Number of concepts (6) and dimensions of confusion matrix do not coincide (5, 5)'
    actual = str(exception).split('ValueError: ')[1]
    assert actual == expected


def test_plot_ROC_curve(metrics_top_k_binary_class, metrics_top_k_multi_class):
    _, y_true_multi, y_probs_multi = metrics_top_k_multi_class
    with pytest.raises(ValueError) as exception:
        plot_ROC_curve(y_probs_multi[:, 1], y_true_multi)
    expected = 'y_true must contain the true binary labels.'
    actual = str(exception).split('ValueError: ')[1]
    assert actual == expected


def test_plot_precision_recall_curve(metrics_top_k_binary_class, metrics_top_k_multi_class):
    _, y_true_multi, y_probs_multi = metrics_top_k_multi_class
    with pytest.raises(ValueError) as exception:
        plot_precision_recall_curve(y_probs_multi[:, 1], y_true_multi)
    expected = 'y_true must contain the true binary labels.'
    actual = str(exception).split('ValueError: ')[1]
    assert actual == expected


def test_plot_concept_metrics():
    metrics = [[0, 0, 0], [0, 0, 0]]
    concepts = ['a', 'b', 'c', 'd', 'e']
    with pytest.raises(ValueError) as exception:
        plot_concept_metrics(concepts, metrics, '', '')
    expected = 'Dimensions of concepts (5) and metrics array (2) do not match'
    actual = str(exception).split('ValueError: ')[1]
    assert actual == expected


def test_plot_threshold():
    th = [0, 0.1, 0.2]
    c = [12, 7, 9, 5]
    e = [2, 5, 10, 6, 6]
    with pytest.raises(ValueError) as exception:
        plot_threshold(th, c, e)
    expected = 'The length of the arrays introduced do not coincide (3), (4), (5)'
    actual = str(exception).split('ValueError: ')[1]
    assert actual == expected


def test_plot_models_performance(test_results_csv_paths):
    with pytest.raises(ValueError) as exception:
        plot_models_performance(eval_dir=test_results_csv_paths, individual=True, class_idx=0, metric=None, save_name='plot.png')
    expected = 'Unsupported type: class_idx, metric'
    actual = str(exception).split('ValueError: ')[1]
    assert actual == expected


def test_plot_confidence_interval():
    values_x = [0, 1]
    values_y = [0, 1]
    lower = [0, 1]
    upper = [0]
    with pytest.raises(ValueError) as exception:
        plot_confidence_interval(values_x, values_y, lower, upper)
    expected = 'Arrays "values_x", "values_y", "lower_bound" and "upper_bound" should have the same dimension'
    actual = str(exception).split('ValueError: ')[1]
    assert actual == expected


def test_scatter_plot():
    values_x = [0, 1]
    values_y = [0]
    with pytest.raises(ValueError) as exception:
        scatter_plot(values_x, values_y, '', '', '')
    expected = 'Both arrays "values_x" and "values_y" should have the same dimension'
    actual = str(exception).split('ValueError: ')[1]
    assert actual == expected
