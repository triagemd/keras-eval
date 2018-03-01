from __future__ import print_function, division, absolute_import
import numpy as np
import scipy
from math import log


def accuracy(y_pred, y_true):
    acc = np.sum(y_true == y_pred) / float(len(y_true))
    return acc


def accuracy_top_k(probs, y_true, k):
    assert k >= 1, "k must be at least 1."
    assert len(probs.shape) == 2, "`probs` should be a matrix of n_samples x n_dimensions"
    assert y_true.shape[0] == probs.shape[0], "The first dimensions should match"
    assert k <= probs.shape[1], "k should be less than the dimension of probs"

    # Sort the predictions. Then get the top k.
    top_k_preds = probs.argsort(axis=1)[:, -k:]

    # Get True/False values for predictions that match.
    preds_match = top_k_preds == y_true[:, np.newaxis]

    # If the row is greater than 0, then at least one of them matched.
    in_top_k = np.sum(preds_match, axis=1) > 0

    # in_top_k is binary, so we can simply sum over how many are positive,
    # divided by how many predictions there are.
    acc = np.sum(in_top_k) / float(len(in_top_k))

    return acc


def metrics_top_k(probs, y_true, class_names, k_vals=(1, 2, 3), verbose=1):
    """
    Compute the sensitivity and precision between the predicted `y_probs` and the true labels `y_true`.

    Notes:
        The precision is only computed when k=1, since it is not clear how to compute when k>1.
        np.nan values are used for undefined values (i.e., divide by zero errors).
            This occurs if `y_probs` has no predictions for a certain class,
            and/or if there is no values for `y_true` for a certain class.

    Args:
        y_true: a list/numpy array of true class labels (*not* encoded as 1-hot).
        y_probs: a numpy array of 1-hot encoded predicted probabilities.
        labels: a list of the names of the labels.
        k_vals: a list of the values of k to use.
        print_screen: boolean that when True prints the results to screen.

    Returns:
        mets: a dictionary of lists of dictionaries containing the results.
            The dict keys correspond to the input `labels`.
            For clarity, see the tests in `tests/test_metrics/test_metrics_top_k()`

    """
    met = []  # {}
    for idx, label in zip(range(len(class_names)), class_names):
        for k in k_vals:
            if k == 0:
                raise ValueError('`k_vals` cannot contain a `0`')

            top_k_preds = probs.argsort(axis=1)[:, -k:]
            preds_match = (top_k_preds == y_true[:, np.newaxis])
            in_top_k = np.sum(preds_match, axis=1) > 0
            tp_top_k = np.sum(in_top_k[(y_true == idx)])

            tp_fn = np.sum(y_true == idx)
            if tp_fn == 0:
                sensitivity = np.nan
            else:
                sensitivity = float(tp_top_k) / tp_fn

            if k == 1:
                y_pred = np.squeeze(top_k_preds)
                tp_fp = np.sum(y_pred == idx)

                if tp_fp == 0:
                    precision = np.nan
                else:
                    precision = float(tp_top_k) / tp_fp

                met.append({'label': label, 'sensitivity_k' + str(k): sensitivity, 'precision_k' + str(k): precision})

                # met[label] = [{'k': k, 'sensitivity': sensitivity, 'precision': precision}]
                if verbose:
                    print('| %s | @k=1, sens:%.3f , prec:%.3f |' % (label.ljust(5), sensitivity, precision),
                          end='')
            else:
                met[idx].update({'sensitivity_k' + str(k): sensitivity})
                # met[label].append({'k': k, 'sens': sensitivity})
                if verbose:
                    print(' @k=%i, sens:%.3f |' % (k, sensitivity), end='')

        if verbose:
            print('')

    return met


def geometric_mean_3D(probs):
    return scipy.stats.gmean(probs, axis=0)


def arithmetic_mean_3D(probs):
    return np.mean(probs, axis=0)


def harmonic_mean_3D(probs):
    return scipy.stats.hmean(probs, axis=0)


def combine_ensemble_probs(probs, combination_mode=None):
    '''
    Args:
        probs: Probailities given by the ensemble of models
        combination_mode: combination_mode: 'arithmetic' / 'geometric' / 'harmonic' mean of the predictions or 'maximum'
           probability value

    Returns: Probabilities combined
    '''
    # Probabilities of the ensemble input=[n_models, n_images, n_class] --> output=[n_images, n_class]

    # Join probabilities given by an ensemble of models following combination mode

    combiners = {
        'arithmetic': np.mean,
        'geometric': scipy.stats.gmean,
        'harmonic': scipy.stats.hmean,
        'maximum': np.amax
    }
    if combination_mode is None:
        raise ValueError('combination_mode is required')
    elif combination_mode not in combiners.keys():
        raise ValueError('Error: invalid option for `combination_mode` ' + str(combination_mode))
    combiner = combiners[combination_mode]
    return combiner(probs, axis=0)


def mean_probability_distribution(probs, verbose=1, combination_mode='arithmetic'):
    '''
    Args:
        probs: Probabilities after model forwarding
        verbose: Show text
        combination_mode: Ensemble combination mode

    Returns: The mean probability for the top-nclasses predictions given

    '''

    # Sort probabilities from high to low, and compute mean
    probs = np.array(probs)
    if probs.ndim > 2:
        prob_mean = np.mean(np.sort(probs)[:, :, ::-1], axis=1)
        prob_mean = combine_ensemble_probs(prob_mean, combination_mode)
    else:
        prob_mean = np.mean(np.sort(probs)[:, ::-1], axis=0)
    if verbose == 1:
        for ind, prob in enumerate(prob_mean):
            print('Confidence mean at giving top %i prediction is %f' % (ind + 1, prob))
    return prob_mean


def uncertainty_dist(probs, verbose=1, combination_mode='arithmetic'):
    '''
    Args:
        probs: Probabilities after model forwarding
        verbose: Show text
        combination_mode: Ensemble combination mode

    Returns: The entropy for each of the predictions given [n_images]
    '''
    probs = np.array(probs)
    # Check if is ensemble
    if probs.ndim > 2:
        probs = combine_ensemble_probs(probs, combination_mode)

    entropy = scipy.stats.entropy(probs.T, base=2.0)

    if verbose == 1:
        print('There are %i classes ' % probs.shape[1])
        print('Max uncertainty value is %.3f' % log(probs.shape[1], 2))
        print('The mean entropy or uncertainty per image is %.3f' % np.mean(entropy))
    return entropy


def get_correct_errors_indices(probs, y_true, k, split_k=False, combination_mode='arithmetic'):
    '''
    Args:
        probs: Probabilities of the model / ensemble [n_images, n_class] / [n_models, n_images, n_class]
        y_true:  Ground truth [n_images, n_class]
        k: Top k probabilities to compute the errors / correct
        split_k: If true, not consider top-n being n<k to compute the top-k correct/error predictions
        combination_mode: For ensembling probabilities

    Returns: Returns A list containing for each of the values of k provided, the indices of the images
            with errors / correct and the probabilities in format [n_images,n_class].
    '''
    probs = np.array(probs)
    if probs.ndim > 2:
        probs = combine_ensemble_probs(probs, combination_mode)

    if k is None:
        raise ValueError('k is required')
    k_list = [k] if not isinstance(k, (list, tuple, np.ndarray)) else k

    errors = []
    correct = []

    # Get ground truth classes
    y_true = y_true.argmax(axis=1)

    for k_val in k_list:

        probs_class = probs.argsort(axis=1)[:, -k_val:]

        # Get True/False values for predictions that match.
        preds_match = probs_class == y_true[:, np.newaxis]

        # Find the positions of the errors / correct
        if split_k:
            # the top-k prediction is the first in each row top-k top-(k-1) ... top-1
            errors.append(np.where(preds_match[:, 0] < 1)[0])
            correct.append(np.where(preds_match[:, 0] > 0)[0])

        else:
            errors.append(np.where(np.sum(preds_match, axis=1) < 1)[0])
            correct.append(np.where(np.sum(preds_match, axis=1) > 0)[0])

    print('Returning correct predictions and errors for the top k: ')
    print(k_list)
    return correct, errors, probs


def get_top1_probability_stats(probs, y_true, threshold, combination_mode='arithmetic', verbose=1):
    '''
    Args:
        probs: Probabilities of the model / ensemble [n_images, n_class] / [n_models, n_images, n_class]
        y_true: Ground truth [n_images, n_class]
        threshold: Value or set of values, can be list, numpy array or tuple
        plot: Show a plot with predicted images / errors in function of th. Can be passed a range of th.
        combination_mode:  For ensembling probabilities
        verbose: Show text

    Returns: A list that for each threshold introduced, has the indices of the images in which we have had an error
    classifying them (top-1) k = 1

    '''
    # Get top-1 errors and correct predictions
    correct, errors, probs = get_correct_errors_indices(probs, y_true, k=1, combination_mode=combination_mode)
    correct = correct[0]
    errors = errors[0]
    # Get the probabilities associated
    error_probabilities = probs[errors]
    correct_probabilities = probs[correct]
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

    return errors_list, correct_list, n_correct, n_errors


def get_top1_entropy_stats(probs, y_true, entropy, combination_mode='arithmetic', verbose=1):
    '''
    Args:
        probs: Probabilities of the model / ensemble [n_images, n_class] / [n_models, n_images, n_class]
        y_true: Ground truth [n_images, n_class]
        plot:  Show a plot with predicted images / errors in function of entropy values
        entropy: Value or set of values, can be list or numpy array with max entropy values
        combination_mode: For ensembling probabilities
        verbose: Show text

    Returns: A list that for each entropy value introduced, has the indices of the images in which we have had an error
    classifying them (top-1) k = 1

    '''
    # Get top-1 errors and correct predictions
    correct, errors, probs = get_correct_errors_indices(probs, y_true, k=1, combination_mode=combination_mode)
    correct = correct[0]
    errors = errors[0]
    # Get the entropy associated
    probs_entropy = uncertainty_dist(probs)
    error_entropy = probs_entropy[errors]
    correct_entropy = probs_entropy[correct]

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

    return errors_list, correct_list, n_correct, n_errors
