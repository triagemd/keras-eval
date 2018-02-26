import numpy as np
from keras_eval.metrics import accuracy_top_k, metrics_top_k


def test_accuracy_top_k():
    y_true = np.asarray([0, 1, 0, 1])
    probs = np.asarray([[1, 0], [0, 1], [1, 0], [0, 1]])
    pred_acc = accuracy_top_k(probs, y_true, k=1)
    assert pred_acc == 1, "All should be true!"

    probs = np.asarray([[0, 1], [1, 0], [0, 1], [1, 0]])
    pred_acc = accuracy_top_k(probs, y_true, k=1)
    assert pred_acc == 0, "All should be False!"

    pred_acc = accuracy_top_k(probs, y_true, k=2)
    assert pred_acc == 1, "All should be True!"

    probs = np.asarray([[0.3, 0.7], [0.4, 0.6], [0.9, 0.1], [0.2, 0.8]])  # First sample an error.
    pred_acc = accuracy_top_k(probs, y_true, k=1)
    assert pred_acc == 0.75, "Accuracy should have 1/4 error."
    pred_acc = accuracy_top_k(probs, y_true k=2)
    assert pred_acc == 1, "Accuracy should have 0 error since k=2."
    # Add an extra class.
    probs = np.asarray([[0.1, 0.7, 0.2], [0.4, 0.6, 0], [0.9, 0.1, 0], [0.2, 0.8, 0]])  # First sample an error.
    pred_acc = accuracy_top_k(probs, y_true, k=2)
    assert pred_acc == 0.75, "Accuracy should have 1/4 error when k=2, since extra class"


def test_metrics_top_k():
    labels = ['class0', 'class1']
    y_true = np.asarray([1, 1, 0])  # 3 samples, 2 classes.
    y_probs = np.asarray([[0, 1], [0.1, 0.9], [0.8, 0.2]])

    met = metrics_top_k(y_probs, y_true, labels, k_vals=[1, 2], print_screen=False)
    assert met[0]['sensitivity_k1'] == 1.0, 'sens should be 1 when k=1.'
    assert met[1]['sensitivity_k1'] == 1.0, 'sens should be 1 when k=1.'
    assert met[0]['precision_k1'] == 1.0, 'prec should be 1 when k=1.'
    assert met[1]['precision_k1'] == 1.0, 'prec should be 1 when k=1.'
    assert met[0]['sensitivity_k2'] == 1.0, 'sens should be 1 when k=2.'
    assert met[1]['sensitivity_k1'] == 1.0, 'sens should be 1 when k=2.'

    y_probs = np.asarray([[1, 0], [0.1, 0.9], [0.8, 0.2]])  # First one is predicted incorrectly.
    met = metrics_top_k(y_probs, y_true, labels, k_vals=[1, 2], print_screen=False)
    assert met[0]['sensitivity_k1'] == 1.0, 'sens should be 1 when k=1, since all correct for class 0.'
    assert met[1]['sensitivity_k1'] == 0.5, 'sens should be 0.5 when k=1, since first is incorrect.'
    assert met[0]['precision_k1'] == 0.5, 'prec should be 0.5 when k=1, since predicts extra class 0.'
    assert met[1]['precision_k1'] == 1.0, 'prec should be 1 when k=1, since does not over predict.'
    assert met[0]['sensitivity_k2'] == 1.0, 'sens should be 1 when k=2.'
    assert met[1]['sensitivity_k2'] == 1.0, 'sens should be 1 when k=2.'

    labels = ['class0', 'class1', 'class2']
    y_probs = np.asarray([[0.1, 0, 0.9], [0.1, 0.9, 0], [0.8, 0.2, 0]])  # Add extra class. First sample is incorrect.
    met = metrics_top_k(y_probs, y_true, labels, k_vals=[1, 2], print_screen=False)
    assert met[0]['sensitivity_k1'] == 1.0, 'sens should be 1 when k=1, since all correct for class 0.'
    assert met[1]['sensitivity_k1'] == 0.5, 'sens should be 0.5 when k=1, since first is incorrect.'
    assert np.isnan(met[2]['sensitivity_k1']), "Should be nan since no values predict class2"
    assert met[2]['precision_k1'] == 0.0, "prec should be 0 since TP is 0."
    assert met[1]['sensitivity_k2'] == 0.5, "When k=2, incorrect first sample (as class0 has higher prob)."
