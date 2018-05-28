import pytest
import numpy as np
from keras_eval.metrics import metrics_top_k, uncertainty_distribution, get_top1_probability_stats, \
    compute_confidence_prediction_distribution, get_correct_errors_indices, get_top1_entropy_stats

from math import log


def test_metrics_top_k():
    concepts = ['class0', 'class1', 'class3']
    y_true = np.asarray([0, 1, 2, 2])  # 4 samples, 3 classes.
    y_probs = np.asarray([[1, 0, 0], [0.2, 0.2, 0.6], [0.8, 0.2, 0], [0.35, 0.25, 0.4]])

    # 2 Correct, 2 Mistakes
    metrics = metrics_top_k(y_probs, y_true, concepts, top_k=1)
    expected = {
        'individual': [
            {'concept': 'class0',
             'metrics': {'precision': [0.5], 'sensitivity': [1.0]}},
            {'concept': 'class1',
             'metrics': {'precision': [np.nan], 'sensitivity': [0.0]}},
            {'concept': 'class3',
             'metrics': {'precision': [0.5], 'sensitivity': [0.5]}}],
        'average': {
            'accuracy': [0.5],
            'confusion_matrix': np.array([[1, 0, 0], [0, 0, 1], [1, 0, 1]]),
            'precision': [0.375],
            'sensitivity': [0.5]}}

    np.testing.assert_equal(metrics, expected)

    # Assert error when top_k <= 0 or > len(concepts)
    with pytest.raises(ValueError) as exception:
        metrics = metrics_top_k(y_probs, y_true, concepts, top_k=0)
    expected = '`top_k` value should be between 1 and the total number of concepts (3)'
    actual = str(exception).split('ValueError: ')[1]
    assert actual == expected

    with pytest.raises(ValueError) as exception:
        metrics = metrics_top_k(y_probs, y_true, concepts, top_k=10)
    expected = '`top_k` value should be between 1 and the total number of concepts (3)'
    actual = str(exception).split('ValueError: ')[1]
    assert actual == expected

    with pytest.raises(ValueError) as exception:
        metrics = metrics_top_k(y_probs, y_true, concepts, top_k=-1)
    expected = '`top_k` value should be between 1 and the total number of concepts (3)'
    actual = str(exception).split('ValueError: ')[1]
    assert actual == expected

    # Assert error when number of samples do not coincide
    y_true = np.asarray([0, 1, 2, 2, 1])
    with pytest.raises(ValueError) as exception:
        metrics = metrics_top_k(y_probs, y_true, concepts, top_k=-1)
    expected = 'The number predicted samples (4) is different from the ground truth samples (5)'
    actual = str(exception).split('ValueError: ')[1]
    assert actual == expected


def test_uncertainty_distribution():
    y_probs = np.array([[0.3, 0.7], [0.67, 0.33]])
    entropy = uncertainty_distribution(y_probs)
    expected_entropy = np.array([0.88, 0.91])
    np.testing.assert_array_equal(np.round(entropy, decimals=2), expected_entropy)


def test_compute_confidence_prediction_distribution():
    y_probs = np.array([[0.3, 0.7], [0.67, 0.33]])
    confidence_prediction = compute_confidence_prediction_distribution(y_probs)
    expected_confidence = np.array([0.68, 0.32])
    np.testing.assert_array_equal(np.round(confidence_prediction, decimals=2), expected_confidence)


def test_get_correct_errors_indices():
    y_probs = np.array([[0.2, 0.8], [0.6, 0.4], [0.9, 0.1]])
    labels = np.array([[0, 1], [0, 1], [1, 0]])

    k = [1]
    correct, errors = get_correct_errors_indices(y_probs, labels, k)
    np.testing.assert_array_equal(correct, [np.array([0, 2])])
    np.testing.assert_array_equal(errors, [np.array([1])])

    # Resilient to k being int
    k = 1
    correct, errors = get_correct_errors_indices(y_probs, labels, k)
    np.testing.assert_array_equal(correct, [np.array([0, 2])])
    np.testing.assert_array_equal(errors, [np.array([1])])

    # multiple k
    k = [1, 2]
    correct, errors = get_correct_errors_indices(y_probs, labels, k)
    np.testing.assert_array_equal(correct[0], np.array([0, 2]))
    np.testing.assert_array_equal(errors[0], np.array([1]))
    np.testing.assert_array_equal(correct[1], np.array([0, 1, 2]))
    np.testing.assert_array_equal(errors[1], np.array([]))


def test_get_top1_entropy_stats():
    y_probs = np.array([[0.2, 0.8], [0.6, 0.4], [0.9, 0.1]])
    labels = np.array([[0, 1], [0, 1], [1, 0]])
    entropy = np.arange(0, log(y_probs.shape[1] + 0.01, 2), 0.1)
    correct_list, errors_list, n_correct, n_errors = get_top1_entropy_stats(y_probs, labels, entropy)

    np.testing.assert_array_equal(n_correct, np.array([0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2]))
    np.testing.assert_array_equal(n_errors, np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]))

    for i, expected_correct in enumerate(n_correct):
        assert len(correct_list[i]) == expected_correct

    for i, expected_errors in enumerate(n_errors):
        assert len(errors_list[i]) == expected_errors

    # One value
    entropy = [0.5]
    correct_list, errors_list, n_correct, n_errors = get_top1_entropy_stats(y_probs, labels, entropy)
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
    correct_list, errors_list, n_correct, n_errors = get_top1_probability_stats(y_probs, labels, threshold)

    np.testing.assert_array_equal(n_correct, np.array([2, 2, 2, 2, 2, 2, 2, 2, 1, 0, 0]))
    np.testing.assert_array_equal(n_errors, np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]))

    for i, expected_correct in enumerate(n_correct):
        assert len(correct_list[i]) == expected_correct

    for i, expected_errors in enumerate(n_errors):
        assert len(errors_list[i]) == expected_errors

    # One value
    threshold = [0.5]
    correct_list, errors_list, n_correct, n_errors = get_top1_probability_stats(y_probs, labels, threshold)
    np.testing.assert_array_equal(n_correct, np.array([2]))
    np.testing.assert_array_equal(n_errors, np.array([1]))

    for i, expected_correct in enumerate(n_correct):
        assert len(correct_list[i]) == expected_correct

    for i, expected_errors in enumerate(n_errors):
        assert len(errors_list[i]) == expected_errors
