import pytest
import numpy as np
import keras_eval.metrics as metrics

from math import log
from collections import OrderedDict


def test_metrics_top_k_multi_class(metrics_top_k_multi_class):
    concepts, y_true, y_probs = metrics_top_k_multi_class

    # 2 Correct, 2 Mistakes for top_k=1
    actual = metrics.metrics_top_k(y_probs, y_true, concepts, top_k=1)
    expected = {
        'individual': [{
            'concept': 'class0',
            'metrics': OrderedDict([
                ('sensitivity', 1.0),
                ('precision', 0.5),
                ('f1_score', 0.6666667),
                ('TP', 1),
                ('FP', 1),
                ('FN', 0),
                ('% of samples', 25.0)])},
            {'concept': 'class1',
             'metrics': OrderedDict([
                 ('sensitivity', 0.0),
                 ('precision', np.nan),
                 ('f1_score', np.nan),
                 ('TP', 0), ('FP', 0),
                 ('FN', 1), ('% of samples', 25.0)])},
            {'concept': 'class3',
             'metrics': OrderedDict([
                 ('sensitivity', 0.5),
                 ('precision', 0.5),
                 ('f1_score', 0.5),
                 ('TP', 1),
                 ('FP', 1),
                 ('FN', 1),
                 ('% of samples', 50.0)])}],
        'average': OrderedDict([
            ('accuracy', 0.5),
            ('weighted_precision', 0.375),
            ('sensitivity', 0.5),
            ('precision', 0.3333333),
            ('f1_score', 0.4166667),
            ('number_of_samples', 4),
            ('number_of_classes', 3),
            ('confusion_matrix', np.array([[1, 0, 0], [0, 0, 1], [1, 0, 1]]))]
        )}

    np.testing.assert_equal(actual, expected)

    # 2 Correct, 2 Mistakes for top_k=2
    actual_accuracy = metrics.metrics_top_k(y_probs, y_true, concepts, top_k=2)['average']['accuracy']
    expected_accuracy = [0.5, 0.75]
    np.testing.assert_equal(actual_accuracy, expected_accuracy)

    actual_sensitivity = metrics.metrics_top_k(y_probs, y_true, concepts, top_k=2)
    actual_sensitivity_class_0 = actual_sensitivity['individual'][0]['metrics']['sensitivity']
    expected_sensitivity_class_0 = [1.0, 1.0]
    np.testing.assert_equal(actual_sensitivity_class_0, expected_sensitivity_class_0)

    actual_sensitivity_class_1 = actual_sensitivity['individual'][1]['metrics']['sensitivity']
    expected_sensitivity_class_1 = [0.0, 1.0]
    np.testing.assert_equal(actual_sensitivity_class_1, expected_sensitivity_class_1)

    actual_sensitivity_class_2 = actual_sensitivity['individual'][2]['metrics']['sensitivity']
    expected_sensitivity_class_2 = [0.5, 0.5]
    np.testing.assert_equal(actual_sensitivity_class_2, expected_sensitivity_class_2)

    # 2 Correct, 2 Mistakes for top_k=3
    actual = metrics.metrics_top_k(y_probs, y_true, concepts, top_k=3)['average']['accuracy']
    expected = [0.5, 0.75, 1.0]
    np.testing.assert_equal(actual, expected)

    actual_sensitivity = metrics.metrics_top_k(y_probs, y_true, concepts, top_k=3)
    actual_sensitivity_class_0 = actual_sensitivity['individual'][0]['metrics']['sensitivity']
    expected_sensitivity_class_0 = [1.0, 1.0, 1.0]
    np.testing.assert_equal(actual_sensitivity_class_0, expected_sensitivity_class_0)

    actual_sensitivity_class_1 = actual_sensitivity['individual'][1]['metrics']['sensitivity']
    expected_sensitivity_class_1 = [0.0, 1.0, 1.0]
    np.testing.assert_equal(actual_sensitivity_class_1, expected_sensitivity_class_1)

    actual_sensitivity_class_2 = actual_sensitivity['individual'][2]['metrics']['sensitivity']
    expected_sensitivity_class_2 = [0.5, 0.5, 1.0]
    np.testing.assert_equal(actual_sensitivity_class_2, expected_sensitivity_class_2)

    # Assert error when top_k <= 0 or > len(concepts)
    with pytest.raises(ValueError) as exception:
        metrics.metrics_top_k(y_probs, y_true, concepts, top_k=0)
    expected = '`top_k` value should be between 1 and the total number of concepts (3)'
    actual = str(exception).split('ValueError: ')[1]
    assert actual == expected

    with pytest.raises(ValueError) as exception:
        metrics.metrics_top_k(y_probs, y_true, concepts, top_k=10)

    expected = '`top_k` value should be between 1 and the total number of concepts (3)'
    actual = str(exception).split('ValueError: ')[1]
    assert actual == expected

    with pytest.raises(ValueError) as exception:
        metrics.metrics_top_k(y_probs, y_true, concepts, top_k=-1)

    expected = '`top_k` value should be between 1 and the total number of concepts (3)'
    actual = str(exception).split('ValueError: ')[1]
    assert actual == expected

    # Assert error when number of samples do not coincide
    y_true = np.asarray([0, 1, 2, 2, 1])
    with pytest.raises(ValueError) as exception:
        metrics.metrics_top_k(y_probs, y_true, concepts, top_k=2)

    expected = 'The number predicted samples (4) is different from the ground truth samples (5)'
    actual = str(exception).split('ValueError: ')[1]
    assert actual == expected


def test_uncertainty_distribution():
    y_probs = np.array([[0.3, 0.7], [0.67, 0.33]])
    entropy = metrics.uncertainty_distribution(y_probs)
    expected_entropy = np.array([0.88, 0.91])
    np.testing.assert_array_equal(np.round(entropy, decimals=2), expected_entropy)


def test_compute_confidence_prediction_distribution():
    y_probs = np.array([[0.3, 0.7], [0.67, 0.33]])
    confidence_prediction = metrics.compute_confidence_prediction_distribution(y_probs)
    expected_confidence = np.array([0.68, 0.32])
    np.testing.assert_array_equal(np.round(confidence_prediction, decimals=2), expected_confidence)


def test_get_correct_errors_indices():
    y_probs = np.array([[0.2, 0.8], [0.6, 0.4], [0.9, 0.1]])
    labels = np.array([[0, 1], [0, 1], [1, 0]])

    k = [1]
    correct, errors = metrics.get_correct_errors_indices(y_probs, labels, k)
    np.testing.assert_array_equal(correct, [np.array([0, 2])])
    np.testing.assert_array_equal(errors, [np.array([1])])

    # Resilient to k being int
    k = 1
    correct, errors = metrics.get_correct_errors_indices(y_probs, labels, k)
    np.testing.assert_array_equal(correct, [np.array([0, 2])])
    np.testing.assert_array_equal(errors, [np.array([1])])

    # multiple k
    k = [1, 2]
    correct, errors = metrics.get_correct_errors_indices(y_probs, labels, k)

    np.testing.assert_array_equal(correct[0], np.array([0, 2]))
    np.testing.assert_array_equal(errors[0], np.array([1]))
    np.testing.assert_array_equal(correct[1], np.array([0, 1, 2]))
    np.testing.assert_array_equal(errors[1], np.array([]))


def test_get_top1_entropy_stats():
    y_probs = np.array([[0.2, 0.8], [0.6, 0.4], [0.9, 0.1]])
    labels = np.array([[0, 1], [0, 1], [1, 0]])

    entropy = np.arange(0, log(y_probs.shape[1] + 0.01, 2), 0.1)
    correct_list, errors_list, n_correct, n_errors = metrics.get_top1_entropy_stats(y_probs, labels, entropy)

    np.testing.assert_array_equal(n_correct, np.array([0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2]))
    np.testing.assert_array_equal(n_errors, np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]))

    for i, expected_correct in enumerate(n_correct):
        assert len(correct_list[i]) == expected_correct

    for i, expected_errors in enumerate(n_errors):
        assert len(errors_list[i]) == expected_errors

    # One value
    entropy = [0.5]
    correct_list, errors_list, n_correct, n_errors = metrics.get_top1_entropy_stats(y_probs, labels, entropy)

    np.testing.assert_array_equal(n_correct, np.array([1]))
    np.testing.assert_array_equal(n_errors, np.array([0]))

    for i, expected_correct in enumerate(n_correct):
        assert len(correct_list[i]) == expected_correct

    for i, expected_errors in enumerate(n_errors):
        assert len(errors_list[i]) == expected_errors


def test_get_top1_probability_stats():
    y_probs = np.array([[0.2, 0.8], [0.6, 0.4], [0.9, 0.1]])
    labels = np.array([[0, 1], [0, 1], [1, 0]])
    threshold = np.arange(0, 1.01, 0.1)

    correct_list, errors_list, n_correct, n_errors = metrics.get_top1_probability_stats(y_probs, labels, threshold)

    np.testing.assert_array_equal(n_correct, np.array([2, 2, 2, 2, 2, 2, 2, 2, 1, 0, 0]))
    np.testing.assert_array_equal(n_errors, np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]))

    for i, expected_correct in enumerate(n_correct):
        assert len(correct_list[i]) == expected_correct

    for i, expected_errors in enumerate(n_errors):
        assert len(errors_list[i]) == expected_errors

    # One value
    threshold = [0.5]
    correct_list, errors_list, n_correct, n_errors = metrics.get_top1_probability_stats(y_probs, labels, threshold)

    np.testing.assert_array_equal(n_correct, np.array([2]))
    np.testing.assert_array_equal(n_errors, np.array([1]))

    for i, expected_correct in enumerate(n_correct):
        assert len(correct_list[i]) == expected_correct

    for i, expected_errors in enumerate(n_errors):
        assert len(errors_list[i]) == expected_errors


def test_confidence_interval_binomial_range():
    values = [0.01, 0.5, 0.99]
    n_samples = [100, 100, 100]

    lower, upper = metrics.confidence_interval_binomial_range(values[0], n_samples[0], 0.95)
    assert lower == 0.0
    assert round(upper, 4) == 0.0295

    lower, upper = metrics.confidence_interval_binomial_range(values[1], n_samples[1], 0.95)
    assert round(lower, 4) == 0.402
    assert round(upper, 4) == 0.598

    lower, upper = metrics.confidence_interval_binomial_range(values[2], n_samples[2], 0.95)
    assert round(lower, 4) == 0.9705
    assert round(upper, 4) == 1.0

    # Assert error when number of samples do not coincide
    with pytest.raises(ValueError) as exception:
        lower, upper = metrics.confidence_interval_binomial_range(values[2], n_samples[2], 0.123)
    expected = 'Confidence value not valid. Confidence values accepted are 0.9, 0.95, 0.98, 0.99 or 90, 95, 98, 99'
    actual = str(exception).split('ValueError: ')[1]
    assert actual == expected


def test_compute_confidence_interval_binomial():
    values_a = np.array([0.01, 0.5, 0.99, 0.02, 0.55])
    values_b = np.array([0.05, 0.05, 0.55])
    probability_interval = np.array([0.01, 0.5, 0.99])
    mean, lower, upper = metrics.compute_confidence_interval_binomial(values_a, values_b,
                                                                      probability_interval=probability_interval)
    lower = [round(l, 4) for l in lower]
    upper = [round(u, 4) for u in upper]

    assert upper == [0.9605, 1.0, 1.0]
    assert lower == [0.2895, 0.3256, 1.0]
    assert mean == [0.625, 0.75, 1.0]


'''
def test_metrics_top_k_binary():
    concepts = ['class0', 'class1']
    y_true = np.asarray([0, 0, 0, 1])  # 4 samples, 2 classes.
    y_probs = np.asarray([[1, 0], [0.75, 0.25], [0.25, 0.75], [0.25, 0.75]])
    # 3 Correct, 1 Mistake
    actual = metrics.metrics_top_k(y_probs, y_true, concepts, top_k=1, round_decimals=3)
    expected = {'individual': [
        {'concept': 'class0',
         'metrics': OrderedDict([
             ('sensitivity', np.float32(0.667)),
             ('precision', 1.0),
             ('f1_score', 0.8),
             ('FDR', 0.0),
             ('AUROC', np.float64(0.667)),
             ('specificity', 1.0),
             ('TP', 2),
             ('FP', 0),
             ('FN', 1),
             ('% of samples', 75.0)])},
        {'concept': 'class1',
         'metrics': OrderedDict([
             ('sensitivity', 1.0),
             ('precision', 0.5),
             ('f1_score', 0.667),
             ('FDR', 0.5),
             ('AUROC', 0.667),
             ('specificity', np.float32(0.667)),
             ('TP', 1),
             ('FP', 1),
             ('FN', 0),
             ('% of samples', 25.0)])}],
        'average': OrderedDict([
            ('accuracy', [0.75]),
            ('precision', 0.875),
            ('f1_score', 0.767),
            ('number_of_samples', 4),
            ('number_of_classes', 2),
            ('confusion_matrix', np.array([[2, 1], [0, 1]]))]
        )}

    np.testing.assert_equal(actual, expected)
'''
