from __future__ import print_function, division, absolute_import
import numpy as np
import scipy.stats
from math import log
import keras_eval.utils as utils
from sklearn.metrics import confusion_matrix


def metrics_top_k(probabilities, ground_truth, concepts, top_k):
    """
    Compute the accuracy, sensitivity and precision between the predicted `probabilities` and the true labels
    `ground_truth`.

    Notes:
        The precision is only computed when k=1, since it is not clear how to compute when k>1.
        np.nan values are used for undefined values (i.e., divide by zero errors).
            This occurs if `y_probabilities` has no predictions for a certain class,
            and/or if there is no values for `ground_truth` for a certain class.

    Args:
        ground_truth: a numpy array of true class labels (*not* encoded as 1-hot).
        probabilities: a numpy array of the predicted probabilities.
        concepts: a list containing the names of the classes.
        top_k: a number specifying the top-k results to compute. E.g. 2 will compute top-1 and top-2

    Returns:
        metrics: a dictionary of lists of dictionaries containing the results.
            The dict keys correspond to the input `labels`.
            For clarity, see the tests in `tests/test_metrics/test_metrics_top_k()`

    """
    if len(probabilities) != len(ground_truth):
        raise ValueError('The number predicted samples (%i) is different from the ground truth samples (%i)' %
                         (len(probabilities), len(ground_truth)))

    if top_k <= 0 or top_k > len(concepts):
        raise ValueError('`top_k` value should be between 1 and the total number of concepts (%i)' % len(concepts))

    top_k = np.arange(1, top_k + 1, 1)
    # Sort predicions from higher to smaller and get class indices
    top_preds = probabilities.argsort(axis=1)[:, ::-1]
    total_samples = ground_truth.shape[0]

    metrics = {'global': {'sensitivity': []}, 'by_concept': []}
    global_accuracy_k = []
    global_precision = []

    for k in top_k:
        global_sensitivity = 0
        # Select top-k predictions
        top_k_preds = top_preds[:, 0:k]
        # Match predictions with ground truth
        matches_k = (top_k_preds == ground_truth[:, np.newaxis])
        in_top_k = np.sum(matches_k, axis=1) > 0
        global_accuracy_k.append(np.sum(in_top_k) / float(len(in_top_k)))

        for idx, concept in zip(range(len(concepts)), concepts):
            tp_top_k = np.sum(in_top_k[(ground_truth == idx)])
            total_samples_concept = np.sum(ground_truth == idx)

            if total_samples_concept == 0:
                sensitivity = np.nan
            else:
                sensitivity = float(tp_top_k) / total_samples_concept
                global_sensitivity += sensitivity * total_samples_concept

            if k == 1:
                total_samples_predicted_as_concept = np.sum(np.squeeze(top_k_preds) == idx)
                if total_samples_predicted_as_concept == 0:
                    precision = np.nan
                else:
                    precision = float(tp_top_k) / total_samples_predicted_as_concept
                    global_precision.append(precision * total_samples_concept)

                metrics['by_concept'].append({
                    'concept': concept,
                    'sensitivity': [sensitivity],
                    'precision': [precision]})

                metrics['global']['confusion_matrix'] = confusion_matrix(ground_truth, top_k_preds,
                                                                         labels=range(len(concepts)))
            else:
                metrics['by_concept'][idx]['sensitivity'].append(sensitivity)

        # Normalize by number of samples
        metrics['global']['sensitivity'].append(global_sensitivity / total_samples)

    metrics['global']['accuracy'] = global_accuracy_k
    metrics['global']['precision'] = [(sum(global_precision) / total_samples)]

    return metrics


def compute_confidence_prediction_distribution(probabilities, combination_mode=None, verbose=1):
    if probabilities.ndim == 3:
        if probabilities.shape[0] <= 1:
            probabilities = probabilities[0]
            prob_mean = np.mean(np.sort(probabilities)[:, ::-1], axis=0)
        else:
            prob_mean = np.mean(np.sort(probabilities)[:, :, ::-1], axis=1)
            prob_mean = utils.combine_probabilities(prob_mean, combination_mode)
    elif probabilities.ndim == 2:
        prob_mean = np.mean(np.sort(probabilities)[:, ::-1], axis=0)
    else:
        raise ValueError('Incorrect shape for `probabilities` array, we accept [n_samples, n_classes] or '
                         '[n_models, n_samples, n_classes]')
    if verbose == 1:
        for ind, prob in enumerate(prob_mean):
            print('Confidence mean at giving top %i prediction is %f' % (ind + 1, prob))

    return prob_mean


def uncertainty_distribution(probabilities, combination_mode=None, verbose=1):
    '''
    Args:
        probabilities: Probabilities after model forwarding
        verbose: Show text
        combination_mode: Ensemble combination mode

    Returns: The entropy for each of the predictions given [n_images]
    '''

    probabilities = utils.combine_probabilities(probabilities, combination_mode)
    entropy = scipy.stats.entropy(probabilities.T, base=2.0)

    if verbose == 1:
        print('There are %i classes ' % probabilities.shape[1])
        print('Max uncertainty value is %.3f' % log(probabilities.shape[1], 2))
        print('The mean entropy or uncertainty per image is %.3f' % np.mean(entropy))
    return entropy


def get_correct_errors_indices(probabilities, labels, k, split_k=False):
    '''
    Args:
        probabilities: Probabilities of the model / ensemble [n_images, n_class] / [n_models, n_images, n_class]
        y_true:  Ground truth [n_images, n_class]
        k: Top k probabilities to compute the errors / correct
        split_k: If true, not consider top-n being n<k to compute the top-k correct/error predictions
        combination_mode: For ensembling probabilities

    Returns: Returns A list containing for each of the values of k provided, the indices of the images
            with errors / correct and the probabilities in format [n_images,n_class].
    '''
    if k is None:
        raise ValueError('k is required')
    k_list = [k] if not isinstance(k, (list, tuple, np.ndarray)) else k

    errors = []
    correct = []

    # Get ground truth classes
    y_true = labels.argmax(axis=1)

    for k_val in k_list:

        probabilities_class = probabilities.argsort(axis=1)[:, -k_val:]

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


def get_top1_probability_stats(probabilities, labels, threshold, verbose=0):
    '''
    Args:
        probabilities: Probabilities of the model / ensemble [n_images, n_class] / [n_models, n_images, n_class]
        y_true: Ground truth [n_images, n_class]
        threshold: Value or set of values, can be list, numpy array or tuple
        combination_mode:  For ensembling probabilities
        verbose: Show text

    Returns: A list that for each threshold introduced, has the indices of the images in which we have had an error
    classifying them (top-1) k = 1

    '''
    # Get top-1 errors and correct predictions
    correct, errors = get_correct_errors_indices(probabilities, labels, k=1)
    correct = correct[0]
    errors = errors[0]
    # Get the probabilities associated
    error_probabilities = probabilities[errors]
    correct_probabilities = probabilities[correct]
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


def get_top1_entropy_stats(probabilities, labels, entropy, verbose=0):
    '''
    Args:
        probabilities: Probabilities of the model / ensemble [n_images, n_class] / [n_models, n_images, n_class]
        y_true: Ground truth [n_images, n_class]
        entropy: Value or set of values, can be list or numpy array with max entropy values
        combination_mode: For ensembling probabilities
        verbose: Show text

    Returns: A list that for each entropy value introduced, has the indices of the images in which we have had an error
    classifying them (top-1) k = 1

    '''
    # Get top-1 errors and correct predictions
    correct, errors = get_correct_errors_indices(probabilities, labels, k=1)
    correct = correct[0]
    errors = errors[0]
    # Get the entropy associated
    probabilities_entropy = uncertainty_distribution(probabilities)
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
