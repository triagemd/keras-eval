from __future__ import print_function
import os
import copy
import numpy as np
import pandas as pd
import keras_eval.utils as utils
import keras_eval.metrics as metrics
import keras_eval.visualizer as visualizer

from math import log


class Evaluator(object):

    OPTIONS = {
        'data_dir': {'type': str, 'default': None},
        'concepts': {'type': list, 'default': None},
        'ensemble_models_dir': {'type': None, 'default': None},
        'model_path': {'type': None, 'default': None},
        'custom_objects': {'type': None, 'default': None},
        'report_dir': {'type': str, 'default': None},
        'combination_mode': {'type': str, 'default': None},
        'id': {'type': str, 'default': 'model'},
        'loss_function': {'type': str, 'default': 'categorical_crossentropy'},
        'metrics': {'type': list, 'default': ['accuracy']},
        'batch_size': {'type': int, 'default': 1},
        'verbose': {'type': int, 'default': 0},
    }

    def __init__(self, **options):
        # Be able to load keras.applications models by default
        self.custom_objects = utils.create_default_custom_objects()

        for key, option in self.OPTIONS.items():
            if key not in options and 'default' not in option:
                raise ValueError('missing required option: %s' % (key, ))
            value = options.get(key, copy.copy(option.get('default')))
            if key == 'custom_objects':
                self.update_custom_objects(value)
            elif key == 'combination_mode':
                self.set_combination_mode(value)
            else:
                setattr(self, key, value)

        extra_options = set(options.keys()) - set(self.OPTIONS.keys())
        if len(extra_options) > 0:
            raise ValueError('unsupported options given: %s' % (', '.join(extra_options), ))

        self.results = None
        self.models = []
        self.model_specs = []
        self.probabilities = None
        self.labels = None

        if self.model_path is not None:
            self.add_model(model_path=self.model_path)

        if self.ensemble_models_dir is not None:
            self.add_model_ensemble(models_dir=self.ensemble_models_dir)

    def update_custom_objects(self, custom_objects):
        if custom_objects is not None and isinstance(custom_objects, dict):
            for key, value in custom_objects.items():
                self.custom_objects.update({key: value})

    def add_model(self, model_path, specs_path=None, custom_objects=None):
        self.update_custom_objects(custom_objects)
        model, model_spec = utils.load_model(model_path=model_path, specs_path=specs_path, custom_objects=self.custom_objects)
        self.models.append(model)
        self.model_specs.append(model_spec)

    def add_model_ensemble(self, models_dir, custom_objects=None):
        self.update_custom_objects(custom_objects)
        models, model_specs = utils.load_multi_model(models_dir=models_dir, custom_objects=self.custom_objects)
        for i, model in enumerate(models):
            self.models.append(model)
            self.model_specs.append(model_specs[i])

    def remove_model(self, model_index):
        self.models.pop(model_index)
        self.model_specs.pop(model_index)

    def set_combination_mode(self, mode):
        modes = ['arithmetic', 'geometric', 'maximum', 'harmonic', None]
        if mode in modes:
            self.combination_mode = mode
        else:
            raise ValueError('Error: invalid option for `combination_mode` ' + str(mode))

    def set_concepts(self, concepts):
        self.concepts = concepts

    def _get_complete_image_paths(self, filenames):
        image_paths = []
        for filename in filenames:
            image_paths.append(os.path.join(self.data_dir, filename))
        return image_paths

    def evaluate(self, data_dir=None, top_k=1, filter_indices=None, confusion_matrix=False,
                 save_confusion_matrix_path=None, verbose=1):
        '''
        Evaluate a set of images. Each sub-folder under 'data_dir/' will be considered as a different class.
        E.g. 'data_dir/class_1/dog.jpg' , 'data_dir/class_2/cat.jpg

        Args:
            data_dir: Data directory to load the images from
            top_k: The top-k predictions to consider. E.g. top_k = 5 is top-5 preds
            filter_indices: If given take only the predictions corresponding to that indices to compute metrics
            confusion_matrix: True/False whether to show the confusion matrix
            save_confusion_matrix_path: If path specified save confusion matrix there
            combination_mode: Ways of combining the model's probabilities to obtain the final prediction.
                'maximum': predictions are obtained by choosing the maximum probabity from each class
                'geometric': predictions are obtained by a geometric mean of all the probabilities
                'arithmetic': predictions are obtained by a arithmetic mean of all the probabilities
                'harmonic': predictions are obtained by a harmonic mean of all the probabilities

        Returns: Probabilities computed and ground truth labels associated.

        '''
        self.data_dir = data_dir or self.data_dir

        if self.data_dir is None:
            raise ValueError('No data directory found, please specify a valid data directory under variable `data_dir`')
        else:
            # Create Keras image generator and obtain predictions
            self.probabilities, self.labels = self._compute_probabilities_generator(data_dir=self.data_dir)

            # Create dictionary containing class names
            if self.concepts is None:
                self.concepts = utils.create_concepts_default(self.num_classes)

            # Obtain labels to show on the metrics results
            self.concept_labels = utils.get_concept_items(self.concepts, key='label')

            # Compute metrics
            self.results = self.get_metrics(probabilities=self.probabilities, labels=self.labels,
                                            concept_labels=self.concept_labels, top_k=top_k, filter_indices=filter_indices,
                                            confusion_matrix=confusion_matrix,
                                            save_confusion_matrix_path=save_confusion_matrix_path)

            if verbose:
                self.print_results(self.results)

        return self.probabilities, self.labels

    def plot_confusion_matrix(self, confusion_matrix, concept_labels=None, save_path=None):
        '''

        Args:
            probabilities: Probabilities from softmax layer
            labels: Ground truth labels
            concept_labels: List containing the class labels
            save_path: If path specified save confusion matrix there

        Returns: Shows the confusion matrix in the screen

        '''
        concept_labels = concept_labels or utils.get_concept_items(self.concepts, key='label')
        visualizer.plot_confusion_matrix(confusion_matrix, concepts=concept_labels, save_path=save_path)

    def get_metrics(self, probabilities, labels, top_k=1, concept_labels=None, filter_indices=None,
                    confusion_matrix=False, save_confusion_matrix_path=None, verbose=0):
        '''
         Print to screen metrics from experiment given probabilities and labels

        Args:
            probabilities: Probabilities from softmax layer
            labels: Ground truth labels
            combination_mode: Ways of combining the model's probabilities to obtain the final prediction.
                'maximum': predictions are obtained by choosing the maximum probability from each class
                'geometric': predictions are obtained by a geometric mean of all the probabilities
                'arithmetic': predictions are obtained by a arithmetic mean of all the probabilities
                'harmonic': predictions are obtained by a harmonic mean of all the probabilities
            K: A tuple of the top-k predictions to consider. E.g. K = (1,2,3,4,5) is top-5 preds
            concept_labels: List containing the concept_labels
            filter_indices: If given take only the predictions corresponding to that indices to compute metrics
            confusion_matrix: If True show the confusion matrix
            save_confusion_matrix_path: If path specified save confusion matrix there
            verbose:

        Returns: Dictionary with metrics for each concept

        '''
        self.combined_probabilities = utils.combine_probabilities(probabilities, self.combination_mode)

        concept_labels = concept_labels or utils.get_concept_items(self.concepts, key='label')

        if filter_indices is not None:
            self.combined_probabilities = self.combined_probabilities[filter_indices]
            labels = labels[filter_indices]

        y_true = labels.argmax(axis=1)

        # Print sensitivity and precision for different values of K.
        results = metrics.metrics_top_k(self.combined_probabilities, y_true, concepts=concept_labels, top_k=top_k)

        # Show metrics visualization as a confusion matrix
        if confusion_matrix:
            self.plot_confusion_matrix(confusion_matrix=results['global']['confusion_matrix'],
                                       concept_labels=concept_labels, save_path=save_confusion_matrix_path)

        return results

    def _compute_probabilities_generator(self, data_dir=None):
        '''

        Args:
            data_dir: Data directory to load the images from

        Returns: Probabilities, ground truth labels of predictions

        '''
        probabilities = []
        if len(self.models) < 1:
            raise ValueError('No models found, please add a valid Keras model first')
        else:
            for i, model in enumerate(self.models):
                print('Making predictions from model ', str(i))
                generator, labels = utils.create_image_generator(data_dir, self.batch_size, self.model_specs[i])
                # N_batches + 1 to gather all the images + collect without repetition [0:n_samples]
                probabilities.append(model.predict_generator(generator=generator,
                                                             steps=(generator.samples // self.batch_size) + 1,
                                                             workers=1,
                                                             verbose=1)[0:generator.samples])

            self.generator = generator
            self.num_classes = generator.num_classes
            self.image_paths = self._get_complete_image_paths(generator.filenames)

            probabilities = np.array(probabilities)

            return probabilities, labels

    def predict(self, data_dir=None):
        '''

        Args:
            data_dir: If folder run _predict_folder, if single image run _predict_image()

        Returns: Probabilities of the folder of images/single image

        '''
        if data_dir is not None:
            self.data_dir = data_dir

        if os.path.isdir(self.data_dir):
            return self._predict_folder(self.data_dir)
        elif self.data_dir.endswith(".png") or self.data_dir.endswith(".jpeg") or self.data_dir.endswith(".jpg"):
            return self._predict_image(self.data_dir)
        else:
            raise ValueError('Wrong data format inputted, please input a valid directory or image path')

    def _predict_folder(self, folder_path):
        '''

        Predict the class probabilities of a set of images from a folder.

        Args:
            folder_path: Path of the folder containing the images

        Returns: Probabilities predicted, image path for every image (aligned with probability)

        '''
        probabilities = []
        for i, model in enumerate(self.models):
            # Read images from folder
            images, image_paths = utils.load_preprocess_images(folder_path, self.model_specs[i])
            images = np.array(images)
            # Predict
            print('Making predictions from model ', str(i))
            probabilities.append(model.predict(images, batch_size=self.batch_size, verbose=1))

        self.probabilities = np.array(probabilities)
        self.combined_probabilities = utils.combine_probabilities(self.probabilities, self.combination_mode)
        self.image_paths = image_paths

        return self.probabilities

    def _predict_image(self, image_path):
        '''

        Predict class probabilities for a single image.

        Args:
            image_path: Path where the image is located

        Returns: Class probabilities for a single image

        '''
        probabilities = []
        for i, model in enumerate(self.models):
            # Read image
            image = utils.load_preprocess_image(image_path, self.model_specs[i])
            # Predict
            print('Making predictions from model ', str(i))
            probabilities.append(model.predict(image, batch_size=1, verbose=1))

        self.probabilities = np.array(probabilities)
        self.combined_probabilities = utils.combine_probabilities(self.probabilities, self.combination_mode)
        self.image_paths = [image_path]

        return self.probabilities

    def show_threshold_impact(self, probabilities, labels, type='probability', threshold=None):
        '''
        Interactive Plot showing the effect of the threshold
        Args:
            probabilities: Probabilities given by the model [n_samples, n_classes]
            labels: Ground truth labels (categorical)
            type: 'Probability' or 'entropy' for a threshold on network top-1 prob or uncertainty in all predictions
            threshold: Custom threshold
            combination_mode: Ways of combining the model's probabilities to obtain the final prediction.
                'maximum': predictions are obtained by choosing the maximum probability from each class
                'geometric': predictions are obtained by a geometric mean of all the probabilities
                'arithmetic': predictions are obtained by a arithmetic mean of all the probabilities
                'harmonic': predictions are obtained by a harmonic mean of all the probabilities

        Returns: The index of the images with error or correct per every threshold, and arrays with the percentage.

        '''
        self.combined_probabilities = utils.combine_probabilities(probabilities, self.combination_mode)

        # Get Error Indices, Number of Correct Predictions, Number of Error Predictions per Threshold
        if type == 'probability':
            threshold = threshold or np.arange(0, 1.01, 0.01)
            correct_ind, errors_ind, correct, errors = metrics.get_top1_probability_stats(self.combined_probabilities, labels,
                                                                                          threshold, verbose=0)
            n_total_errors = errors[0]
            n_total_correct = correct[0]

        elif type == 'entropy':
            threshold = threshold or np.arange(0, log(probabilities.shape[1], 2) + 0.01, 0.01)
            correct_ind, errors_ind, correct, errors = metrics.get_top1_entropy_stats(self.combined_probabilities, labels,
                                                                                      threshold, verbose=0)
            n_total_errors = errors[-1]
            n_total_correct = correct[-1]

        errors = (n_total_errors - errors) / n_total_errors * 100
        correct = correct / n_total_correct * 100

        visualizer.plot_threshold(threshold, correct, errors, title='Top-1 Probability Threshold Tuning')

        return correct_ind, errors_ind, correct, errors

    def get_image_paths_by_prediction(self, probabilities, labels, concept_labels=None, image_paths=None):
        '''

        Args:
            probabilities: Probabilities given by the model [n_samples,n_classes]
            labels: Ground truth labels (categorical)
            concept_labels: List with class names (by default last evaluation)
            image_paths: List with image_paths (by default last evaluation)
            combination_mode: Ways of combining the model's probabilities to obtain the final prediction.
                'maximum': predictions are obtained by choosing the maximum probabity from each class
                'geometric': predictions are obtained by a geometric mean of all the probabilities
                'arithmetic': predictions are obtained by a arithmetic mean of all the probabilities
                'harmonic': predictions are obtained by a harmonic mean of all the probabilities

        Returns: A dictionary containing a list of images per confusion matrix square (relation ClassA_ClassB)

        '''
        self.combined_probabilities = utils.combine_probabilities(probabilities, self.combination_mode)

        if image_paths is None:
            image_paths = self.image_paths

        if self.combined_probabilities.shape[0] != len(image_paths):
            raise ValueError('Length of probabilities (%i) do not coincide with the number of image paths (%i)' %
                             (self.combined_probabilities.shape[0], len(image_paths)))

        concept_labels = concept_labels or utils.get_concept_items(self.concepts, key='label')

        predictions = np.argmax(self.combined_probabilities, axis=1)
        y_true = labels.argmax(axis=1)
        dict_image_paths_concept = {}

        for name_1 in concept_labels:
            for name_2 in concept_labels:
                dict_image_paths_concept.update({name_1 + '_' + name_2: []})

        for i, pred in enumerate(predictions):
            predicted_label = concept_labels[pred]
            correct_label = concept_labels[y_true[i]]
            list_image_paths = dict_image_paths_concept[str(correct_label + '_' + predicted_label)]
            list_image_paths.append(image_paths[i])
            dict_image_paths_concept.update({correct_label + '_' + predicted_label: list_image_paths})

        return dict_image_paths_concept

    def plot_images(self, image_paths, n_imgs=None, title='', save_name=None):
        # Works better defining a number of images between 5 and 30 at a time
        '''

        Args:
            image_paths: List with image_paths
            n_imgs: Number of images to show
            title: Title for the plot

        Returns: Plots images in the screen

        '''
        image_paths = np.array(image_paths)
        if n_imgs is None:
            n_imgs = image_paths.shape[0]

        visualizer.plot_images(image_paths, n_imgs, title, save_name)

    def compute_confidence_prediction_distribution(self, verbose=1):
        '''
        Compute the mean value of the probability assigned to predictions, or how confident is the classifier
        Args:
            combination_mode: Ways of combining the model's probabilities to obtain the final prediction.
                'maximum': predictions are obtained by choosing the maximum probabity from each class
                'geometric': predictions are obtained by a geometric mean of all the probabilities
                'arithmetic': predictions are obtained by a arithmetic mean of all the probabilities
                'harmonic': predictions are obtained by a harmonic mean of all the probabilities
            verbose: Show text

        Returns: The mean value of the probability assigned to predictions [top-1, ..., top-k] k = n_classes

        '''
        if self.probabilities is None:
            raise ValueError('probabilities value is None, please run a evaluation first')
        return metrics.compute_confidence_prediction_distribution(self.probabilities, self.combination_mode, verbose)

    def compute_uncertainty_distribution(self, verbose=1):
        '''
        Compute how the uncertainty is distributed
        Args:
            combination_mode: Ways of combining the model's probabilities to obtain the final prediction.
                'maximum': predictions are obtained by choosing the maximum probabity from each class
                'geometric': predictions are obtained by a geometric mean of all the probabilities
                'arithmetic': predictions are obtained by a arithmetic mean of all the probabilities
                'harmonic': predictions are obtained by a harmonic mean of all the probabilities
            verbose: Show text

        Returns: The uncertainty measurement per each sample

        '''
        if self.probabilities is None:
            raise ValueError('probabilities value is None, please run a evaluation first')
        return metrics.uncertainty_distribution(self.probabilities, self.combination_mode, verbose)

    def plot_top_k_sensitivity_by_concept(self):
        if self.results is None:
            raise ValueError('results parameter is None, please run a evaluation first')
        concepts = utils.get_concept_items(self.concepts, key='label')
        metrics = [item['metrics']['sensitivity'] for item in self.results['by_concept']]
        visualizer.plot_concept_metrics(concepts, metrics, 'Top-k', 'Sensitivity')

    def plot_top_k_accuracy(self):
        if self.results is None:
            raise ValueError('results parameter is None, please run a evaluation first')
        metrics = self.results['global']['accuracy']
        visualizer.plot_concept_metrics(['all'], [metrics], 'Top-k', 'Accuracy')

    @staticmethod
    def print_results(results=None, round_decimals=3, percentage=False):
        if results is None:
            raise ValueError('results parameter is None, please specify a value')
        print('--- Results ---\n')
        print('--- Global Metrics ---\n')
        for key, values in results['global'].items():
            if key != 'confusion_matrix':
                for i in range(0, len(values)):
                    val = round(values[i], round_decimals)
                    if percentage:
                        val = val * 100
                    print('| ' ' @k=' + str(i) + ', ' + key + '=' + '%.3f' % val, end=' ')
                print('|')
        print('')
        print('--- Class Metrics ---\n')
        for concept in results['by_concept']:
            for key_1, val_1 in concept.items():
                if key_1 is 'concept':
                    print('| ' + concept['concept'] + ' ', end='')
                if key_1 is 'metrics':
                    for key_2, val_2 in val_1.items():
                        for i in range(0, len(val_2)):
                            val = round(val_2[i], round_decimals)
                            if percentage:
                                val = val * 100
                            print('| ' ' @k=' + str(i) + ', ' + key_2 + '=' + '%.3f' % val, end=' ')
            print('|')

    def results_to_df(self, csv_path=None, round_decimals=3):
        '''
        Generate a Pandas Dataframe with the results. If csv_path is provided, it writes the Dataframe to a CSV file.
        Args:
            csv_path:
            round_decimals:

        Returns: Pandas Dataframe with results.
        '''
        if self.results is None:
            raise ValueError('results parameter is None, please run a evaluation first')

        df = pd.DataFrame({'model': os.path.basename(self.model_path)}, index=range(1))

        for global_metric in self.results['global']:
            if global_metric is 'confusion_matrix':
                df['TN'] = self.results['global'][global_metric][0, 0]
                df['FN'] = self.results['global'][global_metric][1, 0]
                df['TP'] = self.results['global'][global_metric][1, 1]
                df['FP'] = self.results['global'][global_metric][0, 1]
            elif global_metric is 'precision':
                df[global_metric] = round(self.results['global'][global_metric][0], round_decimals)
            else:
                for k in range(len(self.results['global'][global_metric])):
                    df[global_metric + '_top_' + str(k + 1)] = round(self.results['global'][global_metric][k], round_decimals)

        for concept in self.results['by_concept']:
            for key, value in concept['metrics'].items():
                if key is 'precision':
                    df[concept['concept'] + '_' + str(key)] = round(value[0], round_decimals)
                else:
                    for i in range(0, len(value)):
                        df[concept['concept'] + '_' + str(key) + '_top_' + str(i + 1)] = round(value[i], round_decimals)
        if csv_path:
            df.to_csv(csv_path, index=False)

        return df
