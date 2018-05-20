import pytest
import numpy as np

from math import log
import keras_eval.metrics as metrics


def test_metrics_top_k():
    concepts = ['class0', 'class1', 'class3']
    ground_truth = np.asarray([0, 1, 2, 2])  # 4 samples, 3 classes.
    probabilities = np.asarray([[1, 0, 0], [0.2, 0.2, 0.6], [0.8, 0.2, 0], [0.35, 0.25, 0.4]])

    # 2 Correct, 2 Mistakes
    actual = metrics.metrics_top_k(probabilities, ground_truth, concepts, top_k=1)
    expected = {
        'by_concept': [
            {'concept': 'class0',
             'metrics': {'precision': [0.5], 'sensitivity': [1.0]}},
            {'concept': 'class1',
             'metrics': {'precision': [np.nan], 'sensitivity': [0.0]}},
            {'concept': 'class3',
             'metrics': {'precision': [0.5], 'sensitivity': [0.5]}}],
        'global': {
            'accuracy': [0.5],
            'confusion_matrix': np.array([[1, 0, 0], [0, 0, 1], [1, 0, 1]]),
            'precision': [0.375],
            'sensitivity': [0.5]}}

    np.testing.assert_equal(actual, expected)

    # Assert error when top_k <= 0 or > len(concepts)
    with pytest.raises(ValueError) as exception:
        metrics.metrics_top_k(probabilities, ground_truth, concepts, top_k=0)
    expected = '`top_k` value should be between 1 and the total number of concepts (3)'
    actual = str(exception).split('ValueError: ')[1]
    assert actual == expected

    with pytest.raises(ValueError) as exception:
        metrics.metrics_top_k(probabilities, ground_truth, concepts, top_k=10)
    expected = '`top_k` value should be between 1 and the total number of concepts (3)'
    actual = str(exception).split('ValueError: ')[1]
    assert actual == expected

    with pytest.raises(ValueError) as exception:
        metrics.metrics_top_k(probabilities, ground_truth, concepts, top_k=-1)
    expected = '`top_k` value should be between 1 and the total number of concepts (3)'
    actual = str(exception).split('ValueError: ')[1]
    assert actual == expected

    # Assert error when number of samples do not coincide
    ground_truth = np.asarray([0, 1, 2, 2, 1])
    with pytest.raises(ValueError) as exception:
        metrics.metrics_top_k(probabilities, ground_truth, concepts, top_k=-1)
    expected = 'The number predicted samples (4) is different from the ground truth samples (5)'
    actual = str(exception).split('ValueError: ')[1]
    assert actual == expected


def test_AUROC():
    concepts = ['class0', 'class1', 'class3']
    ground_truth = np.asarray([0, 1, 2, 2])  # 4 samples, 3 classes.
    probabilities = np.asarray([[1, 0, 0], [0.2, 0.2, 0.6], [0.8, 0.2, 0], [0.35, 0.25, 0.4]])
    AUROC_metrics = metrics.AUROC(probabilities, ground_truth, concepts)
    expected_metrics = [{'AUROC': 1.0, 'concept': 'class0'},
                        {'AUROC': 0.5, 'concept': 'class1'},
                        {'AUROC': 0.375, 'concept': 'class3'}]

    assert AUROC_metrics == expected_metrics


def test_average_precision():
    np.testing.assert_array_equal(metrics.average_precision(range(1, 6), [6, 4, 7, 1, 2], 2), 0.25)
    np.testing.assert_array_equal(metrics.average_precision(range(1, 6), [1, 1, 1, 1, 1], 5), 0.2)
    predicted = range(1, 21)
    predicted.extend(range(200, 600))
    np.testing.assert_array_equal(metrics.apk(range(1, 100), predicted, 20), 1.0)


def test_mean_average_precision():
    np.testing.assert_array_equal(metrics.mean_average_precision([range(1, 5)], [range(1, 5)], 3), 1.0)
    np.testing.assert_array_equal(metrics.mean_average_precision([[1, 3, 4], [1, 2, 4], [1, 3]],
                                               [range(1, 6), range(1, 6), range(1, 6)], 3), 0.685185185185185)
    np.testing.assert_array_equal(metrics.mean_average_precision([range(1, 6), range(1, 6)],
                                                                 [[6, 4, 7, 1, 2], [1, 1, 1, 1, 1]], 5), 0.26)
    np.testing.assert_array_equal(metrics.mean_average_precision([[1, 3], [1, 2, 3], [1, 2, 3]],
                                                                 [range(1, 6), [1, 1, 1], [1, 2, 1]], 3), 11.0 / 18)


def test_uncertainty_distribution():
    probabilities = np.array([[0.3, 0.7], [0.67, 0.33]])
    entropy = metrics.uncertainty_distribution(probabilities)
    expected_entropy = np.array([0.88, 0.91])
    np.testing.assert_array_equal(np.round(entropy, decimals=2), expected_entropy)


def test_compute_confidence_prediction_distribution():
    probabilities = np.array([[0.3, 0.7], [0.67, 0.33]])
    confidence_prediction = metrics.compute_confidence_prediction_distribution(probabilities)
    expected_confidence = np.array([0.68, 0.32])
    np.testing.assert_array_equal(np.round(confidence_prediction, decimals=2), expected_confidence)


def test_get_correct_errors_indices():
    probabilities = np.array([[0.2, 0.8], [0.6, 0.4], [0.9, 0.1]])
    labels = np.array([[0, 1], [0, 1], [1, 0]])

    k = [1]
    correct, errors = metrics.get_correct_errors_indices(probabilities, labels, k)
    np.testing.assert_array_equal(correct, [np.array([0, 2])])
    np.testing.assert_array_equal(errors, [np.array([1])])

    # Resilient to k being int
    k = 1
    correct, errors = metrics.get_correct_errors_indices(probabilities, labels, k)
    np.testing.assert_array_equal(correct, [np.array([0, 2])])
    np.testing.assert_array_equal(errors, [np.array([1])])

    # multiple k
    k = [1, 2]
    correct, errors = metrics.get_correct_errors_indices(probabilities, labels, k)
    np.testing.assert_array_equal(correct[0], np.array([0, 2]))
    np.testing.assert_array_equal(errors[0], np.array([1]))
    np.testing.assert_array_equal(correct[1], np.array([0, 1, 2]))
    np.testing.assert_array_equal(errors[1], np.array([]))


def test_get_top1_entropy_stats():
    probabilities = np.array([[0.2, 0.8], [0.6, 0.4], [0.9, 0.1]])
    labels = np.array([[0, 1], [0, 1], [1, 0]])
    entropy = np.arange(0, log(probabilities.shape[1] + 0.01, 2), 0.1)
    correct_list, errors_list, n_correct, n_errors = metrics.get_top1_entropy_stats(probabilities, labels, entropy)

    np.testing.assert_array_equal(n_correct, np.array([0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2]))
    np.testing.assert_array_equal(n_errors, np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]))

    for i, expected_correct in enumerate(n_correct):
        assert len(correct_list[i]) == expected_correct

    for i, expected_errors in enumerate(n_errors):
        assert len(errors_list[i]) == expected_errors

    # One value
    entropy = [0.5]
    correct_list, errors_list, n_correct, n_errors = metrics.get_top1_entropy_stats(probabilities, labels, entropy)
    np.testing.assert_array_equal(n_correct, np.array([1]))
    np.testing.assert_array_equal(n_errors, np.array([0]))

    for i, expected_correct in enumerate(n_correct):
        assert len(correct_list[i]) == expected_correct

    for i, expected_errors in enumerate(n_errors):
        assert len(errors_list[i]) == expected_errors


def test_get_top1_probability_stats():
    probabilities = np.array([[0.2, 0.8], [0.6, 0.4], [0.9, 0.1]])
    labels = np.array([[0, 1], [0, 1], [1, 0]])
    threshold = np.arange(0, 1.01, 0.1)
    correct_list, errors_list, n_correct, n_errors = metrics.get_top1_probability_stats(probabilities, labels, threshold)

    np.testing.assert_array_equal(n_correct, np.array([2, 2, 2, 2, 2, 2, 2, 2, 1, 0, 0]))
    np.testing.assert_array_equal(n_errors, np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]))

    for i, expected_correct in enumerate(n_correct):
        assert len(correct_list[i]) == expected_correct

    for i, expected_errors in enumerate(n_errors):
        assert len(errors_list[i]) == expected_errors

    # One value
    threshold = [0.5]
    correct_list, errors_list, n_correct, n_errors = metrics.get_top1_probability_stats(probabilities, labels, threshold)
    np.testing.assert_array_equal(n_correct, np.array([2]))
    np.testing.assert_array_equal(n_errors, np.array([1]))

    for i, expected_correct in enumerate(n_correct):
        assert len(correct_list[i]) == expected_correct

    for i, expected_errors in enumerate(n_errors):
        assert len(errors_list[i]) == expected_errors
