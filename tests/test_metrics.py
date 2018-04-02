import numpy as np
import keras_eval.metrics as metrics
from math import log


def test_accuracy_top_k():
    y_true = np.asarray([0, 1, 0, 1])
    probs = np.asarray([[1, 0], [0, 1], [1, 0], [0, 1]])
    pred_acc = metrics.accuracy_top_k(probs, y_true, k=1)
    assert pred_acc == 1, "All should be true!"

    probs = np.asarray([[0, 1], [1, 0], [0, 1], [1, 0]])
    pred_acc = metrics.accuracy_top_k(probs, y_true, k=1)
    assert pred_acc == 0, "All should be False!"

    pred_acc = metrics.accuracy_top_k(probs, y_true, k=2)
    assert pred_acc == 1, "All should be True!"

    probs = np.asarray([[0.3, 0.7], [0.4, 0.6], [0.9, 0.1], [0.2, 0.8]])  # First sample an error.
    pred_acc = metrics.accuracy_top_k(probs, y_true, k=1)
    assert pred_acc == 0.75, "Accuracy should have 1/4 error."
    pred_acc = metrics.accuracy_top_k(probs, y_true, k=2)
    assert pred_acc == 1, "Accuracy should have 0 error since k=2."
    # Add an extra class.
    probs = np.asarray([[0.1, 0.7, 0.2], [0.4, 0.6, 0], [0.9, 0.1, 0], [0.2, 0.8, 0]])  # First sample an error.
    pred_acc = metrics.accuracy_top_k(probs, y_true, k=2)
    assert pred_acc == 0.75, "Accuracy should have 1/4 error when k=2, since extra class"


def test_metrics_top_k():
    class_names = ['class0', 'class1']
    y_true = np.asarray([1, 1, 0])  # 3 samples, 2 classes.
    y_probs = np.asarray([[0, 1], [0.1, 0.9], [0.8, 0.2]])

    met = metrics.metrics_top_k(y_probs, y_true, class_names, k_vals=[1, 2], verbose=0)
    assert met[0]['sensitivity_k1'] == 1.0, 'sens should be 1 when k=1.'
    assert met[1]['sensitivity_k1'] == 1.0, 'sens should be 1 when k=1.'
    assert met[0]['precision_k1'] == 1.0, 'prec should be 1 when k=1.'
    assert met[1]['precision_k1'] == 1.0, 'prec should be 1 when k=1.'
    assert met[0]['sensitivity_k2'] == 1.0, 'sens should be 1 when k=2.'
    assert met[1]['sensitivity_k1'] == 1.0, 'sens should be 1 when k=2.'

    y_probs = np.asarray([[1, 0], [0.1, 0.9], [0.8, 0.2]])  # First one is predicted incorrectly.
    met = metrics.metrics_top_k(y_probs, y_true, class_names, k_vals=[1, 2], verbose=0)
    assert met[0]['sensitivity_k1'] == 1.0, 'sens should be 1 when k=1, since all correct for class 0.'
    assert met[1]['sensitivity_k1'] == 0.5, 'sens should be 0.5 when k=1, since first is incorrect.'
    assert met[0]['precision_k1'] == 0.5, 'prec should be 0.5 when k=1, since predicts extra class 0.'
    assert met[1]['precision_k1'] == 1.0, 'prec should be 1 when k=1, since does not over predict.'
    assert met[0]['sensitivity_k2'] == 1.0, 'sens should be 1 when k=2.'
    assert met[1]['sensitivity_k2'] == 1.0, 'sens should be 1 when k=2.'

    class_names = ['class0', 'class1', 'class2']
    y_probs = np.asarray([[0.1, 0, 0.9], [0.1, 0.9, 0], [0.8, 0.2, 0]])  # Add extra class. First sample is incorrect.
    met = metrics.metrics_top_k(y_probs, y_true, class_names, k_vals=[1, 2], verbose=0)
    assert met[0]['sensitivity_k1'] == 1.0, 'sens should be 1 when k=1, since all correct for class 0.'
    assert met[1]['sensitivity_k1'] == 0.5, 'sens should be 0.5 when k=1, since first is incorrect.'
    assert np.isnan(met[2]['sensitivity_k1']), "Should be nan since no values predict class2"
    assert met[2]['precision_k1'] == 0.0, "prec should be 0 since TP is 0."
    assert met[1]['sensitivity_k2'] == 0.5, "When k=2, incorrect first sample (as class0 has higher prob)."


def test_uncertainty_distribution():
    probs = np.array([[0.3, 0.7], [0.67, 0.33]])
    entropy = metrics.uncertainty_distribution(probs)
    expected_entropy = np.array([0.88, 0.91])
    np.testing.assert_array_equal(np.round(entropy, decimals=2), expected_entropy)


def test_get_correct_errors_indices():
    probs = np.array([[0.2, 0.8], [0.6, 0.4], [0.9, 0.1]])
    labels = np.array([[0, 1], [0, 1], [1, 0]])

    k = [1]
    correct, errors = metrics.get_correct_errors_indices(probs, labels, k)
    np.testing.assert_array_equal(correct, [np.array([0, 2])])
    np.testing.assert_array_equal(errors, [np.array([1])])

    # Resilient to k being int
    k = 1
    correct, errors = metrics.get_correct_errors_indices(probs, labels, k)
    np.testing.assert_array_equal(correct, [np.array([0, 2])])
    np.testing.assert_array_equal(errors, [np.array([1])])

    # multiple k
    k = [1, 2]
    correct, errors = metrics.get_correct_errors_indices(probs, labels, k)
    np.testing.assert_array_equal(correct[0], np.array([0, 2]))
    np.testing.assert_array_equal(errors[0], np.array([1]))
    np.testing.assert_array_equal(correct[1], np.array([0, 1, 2]))
    np.testing.assert_array_equal(errors[1], np.array([]))


def test_get_top1_entropy_stats():
    probs = np.array([[0.2, 0.8], [0.6, 0.4], [0.9, 0.1]])
    labels = np.array([[0, 1], [0, 1], [1, 0]])
    entropy = np.arange(0, log(probs.shape[1]+0.01, 2), 0.1)
    correct_list, errors_list, n_correct, n_errors = metrics.get_top1_entropy_stats(probs, labels, entropy)

    np.testing.assert_array_equal(n_correct, np.array([0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2]))
    np.testing.assert_array_equal(n_errors, np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]))

    for i, expected_correct in enumerate(n_correct):
        assert len(correct_list[i]) == expected_correct

    for i, expected_errors in enumerate(n_errors):
        assert len(errors_list[i]) == expected_errors

    # One value
    entropy = [0.5]
    correct_list, errors_list, n_correct, n_errors = metrics.get_top1_entropy_stats(probs, labels, entropy)
    np.testing.assert_array_equal(n_correct, np.array([1]))
    np.testing.assert_array_equal(n_errors, np.array([0]))

    for i, expected_correct in enumerate(n_correct):
        assert len(correct_list[i]) == expected_correct

    for i, expected_errors in enumerate(n_errors):
        assert len(errors_list[i]) == expected_errors


def test_get_top1_probability_stats():
    probs = np.array([[0.2, 0.8], [0.6, 0.4], [0.9, 0.1]])
    labels = np.array([[0, 1], [0, 1], [1, 0]])
    threshold = np.arange(0, 1.01, 0.1)
    correct_list, errors_list, n_correct, n_errors = metrics.get_top1_probability_stats(probs, labels, threshold)

    np.testing.assert_array_equal(n_correct, np.array([2, 2, 2, 2, 2, 2, 2, 2, 1, 0, 0]))
    np.testing.assert_array_equal(n_errors, np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]))

    for i, expected_correct in enumerate(n_correct):
        assert len(correct_list[i]) == expected_correct

    for i, expected_errors in enumerate(n_errors):
        assert len(errors_list[i]) == expected_errors

    # One value
    threshold = [0.5]
    correct_list, errors_list, n_correct, n_errors = metrics.get_top1_probability_stats(probs, labels, threshold)
    np.testing.assert_array_equal(n_correct, np.array([2]))
    np.testing.assert_array_equal(n_errors, np.array([1]))

    for i, expected_correct in enumerate(n_correct):
        assert len(correct_list[i]) == expected_correct

    for i, expected_errors in enumerate(n_errors):
        assert len(errors_list[i]) == expected_errors
