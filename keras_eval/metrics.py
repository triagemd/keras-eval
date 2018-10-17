from __future__ import print_function, division, absolute_import

import numpy as np
import scipy.stats
import keras_eval.utils as utils

from math import log
from sklearn.metrics import confusion_matrix, roc_curve
from keras.utils.np_utils import to_categorical
from collections import OrderedDict


def metrics_top_k(y_probs, y_true, concepts, top_k, round_decimals=7):
    """
    Compute the accuracy, sensitivity and precision between the predicted `y_probs` and the true labels
    `y_true`.

    Notes:
        The precision is only computed when k=1, since it is not clear how to compute when k>1.
        np.nan values are used for undefined values (i.e., divide by zero errors).
            This occurs if `y_probabilities` has no predictions for a certain class,
            and/or if there is no values for `y_true` for a certain class.

    Args:
        y_true: a numpy array of the true class labels (*not* encoded as 1-hot).
        y_probs: a numpy array of the class probabilities.
        concepts: a list containing the names of the classes.
        top_k: a number specifying the top-k results to compute. E.g. 2 will compute top-1 and top-2
        round_decimals: Integer indicating the number of decimals to rounded.

    Returns:
        metrics: a dictionary of lists of dictionaries containing the results.
            The dict keys correspond to the input `labels`.
            For clarity, see the tests in `tests/test_metrics/test_metrics_top_k()`

    """
    if len(y_probs) != len(y_true):
        raise ValueError('The number predicted samples (%i) is different from the ground truth samples (%i)' %
                         (len(y_probs), len(y_true)))

    if top_k <= 0 or top_k > len(concepts):
        raise ValueError('`top_k` value should be between 1 and the total number of concepts (%i)' % len(concepts))

    average_accuracy_k = []
    average_precision = []
    average_f1_score = []

    top_k = np.arange(1, top_k + 1, 1)
    one_hot_y_true = to_categorical(y_true, num_classes=len(concepts))

    # Sort predicions from higher to smaller and get class indices
    top_preds = y_probs.argsort(axis=1)[:, ::-1]
    total_samples = y_true.shape[0]

    metrics = {'average': {}, 'individual': []}

    # top-K Accuracy
    for k in top_k:
        # Select k top predictions
        top_k_preds = top_preds[:, 0:k]
        # Compute the top-k matches
        matches_k = (top_k_preds == y_true[:, np.newaxis])
        in_top_k = np.sum(matches_k, axis=1) > 0
        average_accuracy_k.append(np.sum(in_top_k) / float(len(in_top_k)))

    # Top-1 metrics
    one_hot_top_1_preds = to_categorical(top_preds[:, 0:1], num_classes=len(concepts))

    for idx, concept in enumerate(concepts):
        total_samples_concept = np.sum(y_true == idx)
        percentage_samples_concept = round(total_samples_concept / len(y_true) * 100, 2)

        tn, fp, fn, tp = confusion_matrix(one_hot_y_true[:, idx], one_hot_top_1_preds[:, idx], labels=[0, 1]).astype(
            np.float32).ravel()

        sensitivity = round(utils.safe_divide(tp, tp + fn), round_decimals)

        precision = round(utils.safe_divide(tp, tp + fp), round_decimals)
        if not np.isnan(precision):
            average_precision.append(precision * total_samples_concept)

        f1_score = round(2 * utils.safe_divide(precision * sensitivity, precision + sensitivity), round_decimals)
        if not np.isnan(f1_score):
            average_f1_score.append(f1_score * total_samples_concept)

        metrics_dict = OrderedDict([('sensitivity', sensitivity), ('precision', precision), ('f1_score', f1_score)])

        # Binary classification metrics
        if len(concepts) == 2:
            specificity = round(utils.safe_divide(tn, tn + fp), round_decimals)
            fdr = round(utils.safe_divide(fp, tp + fp), round_decimals)
            fpr, tpr, _ = roc_curve(y_true, y_probs[:, idx], pos_label=idx)
            auroc = round(np.trapz(tpr, fpr))

            metrics_dict.update([('specificity', specificity), ('FDR', fdr), ('AUROC', auroc)])

        metrics_dict.update([('TP', int(tp)), ('FP', int(fp)), ('FN', int(fn)), ('% of samples', percentage_samples_concept)])

        metrics['individual'].append({'concept': concept, 'metrics': metrics_dict})

    accuracy = utils.round_list(average_accuracy_k, round_decimals)

    metrics['average'] = OrderedDict([('accuracy', accuracy if len(accuracy) > 1 else accuracy[0]),
                                      ('precision', round(utils.safe_divide(sum(average_precision), total_samples),
                                                          round_decimals)),
                                      ('f1_score', round(utils.safe_divide(sum(average_f1_score), total_samples),
                                                         round_decimals)),
                                      ('number_of_samples', total_samples),
                                      ('number_of_classes', len(concepts)),
                                      ('confusion_matrix', confusion_matrix(y_true, top_preds[:, 0],
                                                                            labels=np.arange(0, len(concepts))))
                                      ])
    return metrics


def compute_confidence_prediction_distribution(y_probs, combination_mode=None, verbose=1):
    if y_probs.ndim == 3:
        if y_probs.shape[0] <= 1:
            y_probs = y_probs[0]
            prob_mean = np.mean(np.sort(y_probs)[:, ::-1], axis=0)
        else:
            prob_mean = np.mean(np.sort(y_probs)[:, :, ::-1], axis=1)
            prob_mean = utils.combine_probabilities(prob_mean, combination_mode)
    elif y_probs.ndim == 2:
        prob_mean = np.mean(np.sort(y_probs)[:, ::-1], axis=0)
    else:
        raise ValueError('Incorrect shape for `y_probs` array, we accept [n_samples, n_classes] or '
                         '[n_models, n_samples, n_classes]')
    if verbose == 1:
        for ind, prob in enumerate(prob_mean):
            print('Confidence mean at giving top %i prediction is %f' % (ind + 1, prob))

    return prob_mean


def uncertainty_distribution(y_probs, combination_mode=None, verbose=1):
    '''
    Args:
        y_probs: y_probs after model forwarding
        verbose: Show text
        combination_mode: Ensemble combination mode

    Returns: The entropy for each of the predictions given [n_images]
    '''

    y_probs = utils.combine_probabilities(y_probs, combination_mode)
    entropy = scipy.stats.entropy(y_probs.T, base=2.0)

    if verbose == 1:
        print('There are %i classes ' % y_probs.shape[1])
        print('Max uncertainty value is %.3f' % log(y_probs.shape[1], 2))
        print('The mean entropy or uncertainty per image is %.3f' % np.mean(entropy))
    return entropy


def get_correct_errors_indices(y_probs, labels, k, split_k=False):
    '''
    Args:
        y_probs: y_probs of the model / ensemble [n_images, n_class] / [n_models, n_images, n_class]
        y_true:  Ground truth [n_images, n_class]
        k: Top k y_probs to compute the errors / correct
        split_k: If true, not consider top-n being n<k to compute the top-k correct/error predictions
        combination_mode: For ensembling y_probs

    Returns: Returns A list containing for each of the values of k provided, the indices of the images
            with errors / correct and the y_probs in format [n_images,n_class].
    '''
    if k is None:
        raise ValueError('k is required')
    k_list = [k] if not isinstance(k, (list, tuple, np.ndarray)) else k

    errors = []
    correct = []

    # Get ground truth classes
    y_true = labels.argmax(axis=1)

    for k_val in k_list:

        probabilities_class = y_probs.argsort(axis=1)[:, -k_val:]

        # Get True/False values for predictions that match.
        preds_match = probabilities_class == y_true[:, np.newaxis]

        # Find the positions of the errors / correct
        if split_k:
            # the top-k prediction is the first in each row top-k top-(k-1) ... top-1
            errors.append(np.where(preds_match[:, 0] < 1)[0])
            correct.append(np.where(preds_match[:, 0] > 0)[0])
        else:
            errors.append(np.where(np.sum(preds_match, axis=1) < 1)[0])
            correct.append(np.where(np.sum(preds_match, axis=1) > 0)[0])

    print('Returning correct predictions and errors for the top k: ', k_list)
    return correct, errors


def get_top1_probability_stats(y_probs, labels, threshold, verbose=0):
    '''
    Args:
        y_probs: y_probs of the model / ensemble [n_images, n_class] / [n_models, n_images, n_class]
        y_true: Ground truth [n_images, n_class]
        threshold: Value or set of values, can be list, numpy array or tuple
        combination_mode:  For ensembling y_probs
        verbose: Show text

    Returns: A list that for each threshold introduced, has the indices of the images in which we have had an error
    classifying them (top-1) k = 1

    '''
    # Get top-1 errors and correct predictions
    correct, errors = get_correct_errors_indices(y_probs, labels, k=1)
    correct = correct[0]
    errors = errors[0]
    # Get the y_probs associated
    error_probabilities = y_probs[errors]
    correct_probabilities = y_probs[correct]
    # Get the top probability
    top_error_probability = np.sort(error_probabilities, axis=1)[:, ::-1][:, 0]
    top_correct_probability = np.sort(correct_probabilities, axis=1)[:, ::-1][:, 0]

    # If we input a threshold / set of thresholds extract the indices of the images that we will misclassify
    if threshold is None:
        raise ValueError('th is required')
    threshold_list = [threshold] if not isinstance(threshold, (list, tuple, np.ndarray)) else threshold

    # Compute indices of the errors over certain th
    errors_list = []
    # Compute indices of the correct over certain th
    correct_list = []

    # number of correct / error predictions
    n_correct = []
    n_errors = []

    for i, thres in enumerate(threshold_list):

        errors_over_th = [errors[i] for i, prob in enumerate(top_error_probability) if prob > thres]
        correct_over_th = [correct[i] for i, prob in enumerate(top_correct_probability) if prob > thres]

        errors_over_th = np.array(errors_over_th)
        correct_over_th = np.array(correct_over_th)
        n_correct.append(correct_over_th.shape[0])
        n_errors.append(errors_over_th.shape[0])
        errors_list.append(errors_over_th)
        correct_list.append(correct_over_th)

        if verbose == 1:
            print('-- For a threshold value of  %.2f --\n' % thres)
            print('The total number of errors was %i\nThere were %i errors that will be classified as correct\n'
                  % (errors.shape[0], errors_over_th.shape[0]))

            print('%.3f%% of the errors will be removed\n'
                  % (((errors.shape[0] - errors_over_th.shape[0]) / errors.shape[0]) * 100))

            print('The total number of correct predictions was %i\nThere are %i that will be predicted as correct\n'
                  % (correct.shape[0], correct_over_th.shape[0]))

            print('%.3f%% of the correct predictions will be predicted\n'
                  % ((correct_over_th.shape[0] / correct.shape[0]) * 100))

    n_correct = np.array(n_correct)
    n_errors = np.array(n_errors)

    return correct_list, errors_list, n_correct, n_errors


def get_top1_entropy_stats(y_probs, labels, entropy, verbose=0):
    '''
    Args:
        y_probs: y_probs of the model / ensemble [n_images, n_class] / [n_models, n_images, n_class]
        y_true: Ground truth [n_images, n_class]
        entropy: Value or set of values, can be list or numpy array with max entropy values
        combination_mode: For ensembling y_probs
        verbose: Show text

    Returns: A list that for each entropy value introduced, has the indices of the images in which we have had an error
    classifying them (top-1) k = 1

    '''
    # Get top-1 errors and correct predictions
    correct, errors = get_correct_errors_indices(y_probs, labels, k=1)
    correct = correct[0]
    errors = errors[0]
    # Get the entropy associated
    probabilities_entropy = uncertainty_distribution(y_probs)
    error_entropy = probabilities_entropy[errors]
    correct_entropy = probabilities_entropy[correct]

    # If we input a threshold / set of thresholds extract the indices of the images that we will misclassify
    if entropy is None:
        raise ValueError('ent is required')
    entropy_list = [entropy] if not isinstance(entropy, (list, tuple, np.ndarray)) else entropy

    # Compute indices of the errors over certain entropy
    errors_list = list()
    # Compute indices of the correct over certain entropy
    correct_list = list()

    # number of correct / error predictions
    n_correct = []
    n_errors = []

    for i, ent in enumerate(entropy_list):
        errors_below_e = [errors[i] for i, entropy_value in enumerate(error_entropy) if entropy_value < ent]

        correct_below_e = [correct[i] for i, entropy_value in enumerate(correct_entropy) if entropy_value < ent]

        errors_below_e = np.array(errors_below_e)
        correct_below_e = np.array(correct_below_e)
        n_correct.append(correct_below_e.shape[0])
        n_errors.append(errors_below_e.shape[0])

        errors_list.append(errors_below_e)
        correct_list.append(correct_below_e)

        if verbose == 1:
            print('-- For a entropy value of  %.2f --\n' % ent)
            print('The total number of errors was %i\nThere were %i errors that will be classified as correct\n'
                  % (errors.shape[0], errors_below_e.shape[0]))

            print('%.3f%% of the errors will be removed\n'
                  % (((errors.shape[0] - errors_below_e.shape[0]) / errors.shape[0]) * 100))

            print('The total number of correct predictions was %i\nThere are %i that will be predicted as correct\n'
                  % (correct.shape[0], correct_below_e.shape[0]))

            print('%.3f%% of the correct predictions will be predicted\n'
                  % ((correct_below_e.shape[0] / correct.shape[0]) * 100))

    n_correct = np.array(n_correct)
    n_errors = np.array(n_errors)

    return correct_list, errors_list, n_correct, n_errors
