import os
import copy
import numpy as np
import keras_eval.utils as utils
import keras_eval.metrics as metrics
import keras_eval.visualizer as visualizer

from math import log
from keras.utils import generic_utils


class Evaluator(object):
    OPTIONS = {
        'data_dir': {'type': str, 'default': None},
        'concepts': {'type': list, 'default': None},
        'ensemble_models_dir': {'type': None, 'default': None},
        'model_path': {'type': None, 'default': None},
        'custom_objects': {'type': None, 'default': None},
        'report_dir': {'type': str, 'default': None},
        'combination_mode': {'type': str, 'default': None},
        'id': {'type': str, 'default': None},
        'concept_dictionary_path': {'type': str, 'default': None},
        'loss_function': {'type': str, 'default': 'categorical_crossentropy'},
        'metrics': {'type': list, 'default': ['accuracy']},
        'batch_size': {'type': int, 'default': 1},
        'verbose': {'type': int, 'default': 0},
        'data_augmentation': {'type': dict, 'default': None},
    }

    def __init__(self, **options):
        # Be able to load Keras_applications models by default
        self.custom_objects = utils.create_default_custom_objects()

        for key, option in self.OPTIONS.items():
            if key not in options and 'default' not in option:
                raise ValueError('missing required option: %s' % (key,))
            value = options.get(key, copy.copy(option.get('default')))
            if key == 'custom_objects':
                self.update_custom_objects(value)
            elif key == 'combination_mode':
                self.set_combination_mode(value)
            elif key == 'concept_dictionary_path' and options.get('concept_dictionary_path') is not None:
                self.concept_dictionary = utils.read_dictionary(value)
            else:
                setattr(self, key, value)
            if key == 'id' and options.get('model_path') is not None:
                if value is None:
                    self.id = os.path.basename(options.get('model_path'))

        extra_options = set(options.keys()) - set(self.OPTIONS.keys())
        if len(extra_options) > 0:
            raise ValueError('unsupported options given: %s' % (', '.join(extra_options),))

        self.results = None
        self.models = []
        self.model_specs = []
        self.probabilities = None
        self.labels = None
        self.group_id_dict = {}

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
        model, model_spec = utils.load_model(model_path=model_path, specs_path=specs_path,
                                             custom_objects=self.custom_objects)
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
        for concept_dict in concepts:
            if 'label' not in concept_dict.keys() and 'id' not in concept_dict.keys():
                raise ValueError('Incorrect format for concepts list. It must contain the fields `id` and `label`')
        self.concepts = concepts

    def _get_complete_image_paths(self, filenames):
        image_paths = []
        for filename in filenames:
            image_paths.append(os.path.join(self.data_dir, filename))
        return image_paths

    def evaluate(self, data_dir=None, top_k=1, filter_indices=None, confusion_matrix=False, data_augmentation=None,
                 save_confusion_matrix_path=None, show_confusion_matrix_text=True):
        '''
        Evaluate a set of images. Each sub-folder under 'data_dir/' will be considered as a different class.
        E.g. 'data_dir/class_1/dog.jpg' , 'data_dir/class_2/cat.jpg

        Args:
            data_dir: Data directory to load the images from
            top_k: The top-k predictions to consider. E.g. top_k = 5 is top-5 preds
            filter_indices: If given take only the predictions corresponding to that indices to compute metrics
            confusion_matrix: True/False whether to show the confusion matrix
            It includes the addition of data_augmentation as an argument. It is a dictionary consisting of 3 elements:
            - 'scale_sizes': 'default' (4 scales similar to Going Deeper with Convolutions work) or a list of sizes.
            Each scaled image then will be cropped into three square parts.
            - 'transforms': list of transforms to apply to these crops in addition to not
            applying any transform ('horizontal_flip', 'vertical_flip', 'rotate_90', 'rotate_180', 'rotate_270' are
            supported now).
            - 'crop_original': 'center_crop' mode allows to center crop the original image prior do the rest of
            transforms, scalings + croppings.

            save_confusion_matrix_path: If path specified save confusion matrix there


        Returns: Probabilities computed and ground truth labels associated.

        '''
        self.top_k = top_k
        self.data_dir = data_dir or self.data_dir
        self.data_augmentation = data_augmentation or self.data_augmentation
        if self.data_dir is None:
            raise ValueError('No data directory found, please specify a valid data directory under variable `data_dir`')
        else:
            # Create dictionary containing class names
            if self.concepts is None:
                self.concepts = utils.get_default_concepts(self.data_dir)

            # Obtain labels to show on the metrics results
            self.concept_labels = utils.get_concept_items(self.concepts, key='label')

            if hasattr(self, 'concept_dictionary'):
                if utils.compare_group_test_concepts(self.concept_labels, self.concept_dictionary) \
                        and utils.check_concept_unique(self.concept_dictionary):
                    # Create Keras image generator and obtain probabilities
                    self.probabilities, self.labels = self._compute_probabilities_generator(
                        data_dir=self.data_dir, data_augmentation=self.data_augmentation)
                    self.compute_inference_probabilities(self.probabilities)

            else:
                # Create Keras image generator and obtain probabilities
                self.probabilities, self.labels = self._compute_probabilities_generator(
                    data_dir=self.data_dir, data_augmentation=self.data_augmentation)

            # Compute metrics
            self.results = self.get_metrics(probabilities=self.probabilities, labels=self.labels,
                                            concept_labels=self.concept_labels, top_k=top_k,
                                            filter_indices=filter_indices,
                                            confusion_matrix=confusion_matrix,
                                            save_confusion_matrix_path=save_confusion_matrix_path,
                                            show_confusion_matrix_text=show_confusion_matrix_text)

        return self.probabilities, self.labels

    def save_probabilities_labels(self, id, save_path):
        if self.combined_probabilities is not None and self.labels is not None:
            utils.save_numpy(id + '_probabilities', save_path, self.combined_probabilities)
            utils.save_numpy(id + '_labels', save_path, self.labels)

    def compute_inference_probabilities(self, probabilities):
        '''
        Computes the class probability inference based on key "group" in concept_dictionary and saves the values in
        self.probabilities

        Args:
            probabilities: Class inference probabilities with shape [model,samples,inferred_classes].

        '''
        for concept in self.concept_dictionary:
            if concept['group'] in self.group_id_dict.keys():
                self.group_id_dict[concept['group']].append(concept['class_index'])
            else:
                self.group_id_dict[concept['group']] = [concept['class_index']]

        inference_probabilities = []
        for model in range(len(probabilities)):
            single_inference_probabilities = np.zeros((len(probabilities[0]), len(self.concept_labels)))
            for idx, concept_label in enumerate(self.concept_labels):
                column_numbers = self.group_id_dict[concept_label]
                for column_number in column_numbers:
                    single_inference_probabilities[:, idx] += probabilities[model][:, column_number]
            inference_probabilities.append(single_inference_probabilities)

        self.probabilities = np.array(inference_probabilities)

    def plot_confusion_matrix(self, confusion_matrix, concept_labels=None, save_path=None, show_text=True,
                              show_labels=True):
        '''

        Args:
            probabilities: Probabilities from softmax layer
            labels: Ground truth labels
            concept_labels: List containing the class labels
            save_path: If path specified save confusion matrix there

        Returns: Shows the confusion matrix in the screen

        '''
        concept_labels = concept_labels or utils.get_concept_items(self.concepts, key='label')
        visualizer.plot_confusion_matrix(confusion_matrix, concepts=concept_labels, save_path=save_path,
                                         show_text=show_text, show_labels=show_labels)

    def get_metrics(self, probabilities, labels, top_k=1, concept_labels=None, filter_indices=None,
                    confusion_matrix=False, save_confusion_matrix_path=None, show_confusion_matrix_text=True):
        '''
        Print to screen metrics from experiment given probabilities and labels

        Args:
            probabilities: Probabilities from softmax layer
            labels: Ground truth labels
            K: A tuple of the top-k predictions to consider. E.g. K = (1,2,3,4,5) is top-5 preds
            concept_labels: List containing the concept_labels
            filter_indices: If given take only the predictions corresponding to that indices to compute metrics
            confusion_matrix: If True show the confusion matrix
            save_confusion_matrix_path: If path specified save confusion matrix there

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
            self.plot_confusion_matrix(confusion_matrix=results['average']['confusion_matrix'],
                                       concept_labels=concept_labels, save_path=save_confusion_matrix_path,
                                       show_text=show_confusion_matrix_text,
                                       show_labels=show_confusion_matrix_text)

        return results

    def _compute_probabilities_generator(self, data_dir=None, data_augmentation=None):
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

                if data_augmentation is None:
                    generator, labels = utils.create_image_generator(data_dir, self.batch_size, self.model_specs[i])
                    # N_batches + 1 to gather all the images + collect without repetition [0:n_samples]
                    probabilities.append(model.predict_generator(generator=generator,
                                                                 steps=(generator.samples // self.batch_size) + 1,
                                                                 workers=1,
                                                                 verbose=1)[0:generator.samples])
                else:
                    generator, labels = utils.create_image_generator(data_dir, self.batch_size, self.model_specs[i],
                                                                     data_augmentation=data_augmentation)
                    print('Averaging probabilities of %i different outputs at sizes: %s with transforms: %s'
                          % (generator.n_crops, generator.scale_sizes, generator.transforms))
                    steps = generator.samples
                    probabilities_model = []
                    for k, batch in enumerate(generator):
                        if k == steps:
                            break
                        progbar = generic_utils.Progbar(steps)
                        progbar.add(k + 1)
                        probs = model.predict(batch[0][0], batch_size=self.batch_size)
                        probabilities_model.append(np.mean(probs, axis=0))
                    probabilities.append(probabilities_model)

            self.generator = generator
            self.num_classes = generator.num_classes
            self.image_paths = self._get_complete_image_paths(generator.filenames)

            probabilities = np.array(probabilities)

            return probabilities, labels

    def predict(self, data_dir=None, image_list=None, verbose=True):
        '''

        Args:
            data_dir: If folder run _predict_folder, if single image run _predict_image()

        Returns: Probabilities of the folder of images/single image

        '''
        if data_dir is not None:
            self.data_dir = data_dir

        if image_list is not None and isinstance(image_list, list):
            return self._predict_list(image_list, verbose=verbose)
        elif os.path.isdir(self.data_dir):
            return self._predict_folder(self.data_dir, verbose=verbose)
        elif self.data_dir.endswith(".png") or self.data_dir.endswith(".jpeg") or self.data_dir.endswith(".jpg"):
            return self._predict_image(self.data_dir, verbose=verbose)
        else:
            raise ValueError('Wrong data format inputted, please input a valid directory or image path')

    def _predict_folder(self, folder_path, verbose=True):
        '''

        Predict the class probabilities of a set of images from a folder.

        Args:
            folder_path: Path of the folder containing the images

        Returns: Probabilities predicted

        '''
        probabilities = []
        for i, model in enumerate(self.models):
            # Read images from folder
            images, image_paths = utils.load_preprocess_images(folder_path, self.model_specs[i])
            images = np.array(images)
            # Predict
            if verbose:
                print('Making predictions from model ', str(i))
            probabilities.append(model.predict(images, batch_size=self.batch_size, verbose=verbose))

        self.probabilities = np.array(probabilities)
        self.combined_probabilities = utils.combine_probabilities(self.probabilities, self.combination_mode)
        self.image_paths = image_paths

        return self.probabilities

    def _predict_list(self, image_list, verbose=True):
        '''

        Predict the class probabilities of a set of images from a given list.

        Args:
            image_list: List of image paths

        Returns: Probabilities predicted

        '''
        probabilities = []
        image_paths = []

        for image_path in image_list:
            # Read images from folder
            probabilities.append(self._predict_image(image_path, verbose)[0])
            image_paths.append(image_path)

        self.probabilities = np.array(probabilities)
        self.image_paths = image_paths

        return self.probabilities

    def _predict_image(self, image_path, verbose):
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
            if verbose:
                print('Making predictions from model ', str(i))
            probabilities.append(model.predict(image, batch_size=1, verbose=verbose))

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
            correct_ind, errors_ind, correct, errors = metrics.get_top1_probability_stats(self.combined_probabilities,
                                                                                          labels,
                                                                                          threshold, verbose=0)
            n_total_errors = errors[0]
            n_total_correct = correct[0]

        elif type == 'entropy':
            threshold = threshold or np.arange(0, log(probabilities.shape[1], 2) + 0.01, 0.01)
            correct_ind, errors_ind, correct, errors = metrics.get_top1_entropy_stats(self.combined_probabilities,
                                                                                      labels,
                                                                                      threshold, verbose=0)
            n_total_errors = errors[-1]
            n_total_correct = correct[-1]

        errors = (n_total_errors - errors) / n_total_errors * 100
        correct = correct / n_total_correct * 100

        visualizer.plot_threshold(threshold, correct, errors, title='Top-1 Probability Threshold Tuning')

        return correct_ind, errors_ind, correct, errors

    def get_image_paths_by_prediction(self, probabilities, labels, concept_labels=None, image_paths=None):
        '''
        Return the list of images given its predictions.
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

        Returns: A dictionary containing a list of images per confusion matrix square (relation ClassA_ClassB), and the
        predicted probabilities

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
                if name_1 == name_2:
                    dict_image_paths_concept.update({name_1 + '_' + name_2: {'image_paths': [], 'probs': [],
                                                                             'diagonal': True}})
                else:
                    dict_image_paths_concept.update({name_1 + '_' + name_2: {'image_paths': [], 'probs': [],
                                                                             'diagonal': False}})

        for i, pred in enumerate(predictions):
            predicted_label = concept_labels[pred]
            correct_label = concept_labels[y_true[i]]
            list_image_paths = dict_image_paths_concept[str(correct_label + '_' + predicted_label)]['image_paths']
            list_image_paths.append(image_paths[i])
            list_probs = dict_image_paths_concept[str(correct_label + '_' + predicted_label)]['probs']
            list_probs.append(self.combined_probabilities[i])
            diagonal = dict_image_paths_concept[str(correct_label + '_' + predicted_label)]['diagonal']
            dict_image_paths_concept.update({correct_label + '_' + predicted_label:
                                             {
                                                 'image_paths': list_image_paths,
                                                 'probs': list_probs,
                                                 'diagonal': diagonal
                                             }
                                             })

        return dict_image_paths_concept

    @staticmethod
    def plot_images(image_paths, n_images=None, title='', n_cols=5, image_res=(20, 20), save_name=None):
        # Works better defining a number of images between 5 and 30 at a time
        '''

        Args:
            image_paths: List with image_paths
            n_images: Number of images to show
            title: Title for the plot
            n_cols: Number of columns to split the data
            image_res: Plot image resolution
            save_name: If specified, will save the plot in save_name path

        Returns: Plots images in the screen

        '''

        image_paths = np.array(image_paths)
        if n_images is None:
            n_images = image_paths.shape[0]

        visualizer.plot_images(image_paths, n_images, title, None, n_cols, image_res, save_name)

    def plot_probability_histogram(self, mode='errors', bins=100):
        '''

        Args:
            mode: Two modes, "correct" and "error" are supported
            bins: Number of histogram bins

        Returns:

        '''
        if self.probabilities is None:
            raise ValueError('There are not computed probabilities. Please run an evaluation first.')

        self.combined_probabilities = utils.combine_probabilities(self.probabilities, self.combination_mode)
        correct, errors = metrics.get_correct_errors_indices(self.combined_probabilities, self.labels, k=1)
        probs_top = np.max(self.combined_probabilities, axis=1)

        if mode == 'errors':
            probs = probs_top[errors[0]]
        elif mode == 'correct':
            probs = probs_top[correct[0]]
        else:
            raise ValueError('Incorrect mode. Supported modes are "errors" and "correct"')

        visualizer.plot_histogram(probs, bins, 'Histogram of ' + mode + ' probabilities', 'Probability', '')

    def plot_most_confident(self, mode='errors', title='', n_cols=5, n_images=None, image_res=(20, 20), save_name=None):
        '''
    Plots most confident errors or correct detections
        Args:
            mode: Two modes, "correct" and "error" are supported
            title: Title of the Plot
            n_cols: Number of columns
            n_images: Number of images to show
            image_res: Plot image resolution
            save_name: If specified, will save the plot in save_name path

        Returns: Sorted image paths with corresponding probabilities

        '''
        if self.probabilities is None:
            raise ValueError('There are not computed probabilities. Please run an evaluation first.')

        self.combined_probabilities = utils.combine_probabilities(self.probabilities, self.combination_mode)
        correct, errors = metrics.get_correct_errors_indices(self.combined_probabilities, self.labels, k=1)
        probs_top = np.max(self.combined_probabilities, axis=1)

        if mode == 'errors':
            probs = probs_top[errors[0]]
        elif mode == 'correct':
            probs = probs_top[correct[0]]
        else:
            raise ValueError('Incorrect mode. Supported modes are "errors" and "correct"')

        image_paths = np.array(self.image_paths)
        index_max = np.argsort(probs)[::-1]
        image_paths = image_paths[index_max]

        if n_images is None:
            n_images = min(len(image_paths), 20)

        subtitles = ['Prob=' + str(prob)[0:5] for prob in probs[index_max]][0:n_images]

        visualizer.plot_images(image_paths, n_images, title, subtitles[0:n_images], n_cols, image_res, save_name)

        return image_paths, probs[index_max]

    def plot_confidence_interval(self, mode='accuracy', confidence_value=0.95,
                                 probability_interval=np.arange(0, 1.0, 0.01)):
        '''
        Computes and plot the confidence interval for a given mode. It uses a confidence value for a given success and
        failure values following a binomial distribution using the gaussian approximation.
        Args:
            mode: Two modes, "accuracy" and "error" are supported
            confidence_value:  Percentage of confidence. Values accepted are 0.9, 0.95, 0.98, 0.99 or 90, 95, 98, 99
            probability_interval: Probabilities to compare with.

        Returns: Mean, lower and upper bounds for each probability. Plot the graph.

        '''
        if self.probabilities is None:
            raise ValueError('There are not computed probabilities. Please run an evaluation first.')

        self.combined_probabilities = utils.combine_probabilities(self.probabilities, self.combination_mode)
        correct, errors = metrics.get_correct_errors_indices(self.combined_probabilities, self.labels, k=1)
        probs_correct = np.max(self.combined_probabilities, axis=1)[correct[0]]
        probs_error = np.max(self.combined_probabilities, axis=1)[errors[0]]

        if mode == 'accuracy':
            title = 'Accuracy Confidence Interval'
            mean, lower_bound, upper_bound = \
                metrics.compute_confidence_interval_binomial(probs_correct, probs_error,
                                                             confidence_value, probability_interval)
        elif mode == 'error':
            title = 'Error Confidence Interval'
            mean, lower_bound, upper_bound = \
                metrics.compute_confidence_interval_binomial(probs_error, probs_correct,
                                                             confidence_value, probability_interval)
        else:
            raise ValueError('Incorrect mode. Modes available are "accuracy" or "error".')

        visualizer.plot_confidence_interval(probability_interval, mean, lower_bound, upper_bound, title=title)
        return mean, lower_bound, upper_bound

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
            raise ValueError('probabilities value is None, please run an evaluation first')
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
            raise ValueError('probabilities value is None, please run an evaluation first')
        return metrics.uncertainty_distribution(self.probabilities, self.combination_mode, verbose)

    def plot_top_k_sensitivity_by_concept(self):
        if self.results is None:
            raise ValueError('results parameter is None, please run an evaluation first')
        concepts = utils.get_concept_items(self.concepts, key='label')
        metrics = [item['metrics']['sensitivity'] for item in self.results['individual']]
        visualizer.plot_concept_metrics(concepts, metrics, 'Top-k', 'Sensitivity')

    def plot_top_k_accuracy(self):
        if self.results is None:
            raise ValueError('results parameter is None, please run an evaluation first')
        metrics = self.results['average']['accuracy']
        visualizer.plot_concept_metrics(['all'], [metrics], 'Top-k', 'Accuracy')

    def show_results(self, mode='average', round_decimals=3, show_id=True):
        '''
        Args:
            mode: Mode of results. "average" will show the average metrics while "individual" will show metrics by class
            round_decimals: Decimal position to round the numbers.
            show_id: Show id in the first column.

        Returns: Pandas dataframe with results.

        '''
        if self.results is None:
            raise ValueError('results parameter is None, please run an evaluation first')

        return utils.results_to_dataframe(self.results, self.id, mode, round_decimals, show_id)

    def save_results(self, id, csv_path, mode='average', round_decimals=3, show_id=True):
        '''

        Args:
            id: Name of the results evaluation
            csv_path: If specified, results will be saved on that location
            mode: Mode of results. "average" will show the average metrics while "individual" will show metrics by class
            round_decimals: Decimal position to round the numbers.
            show_id: Show id in the first column.

        Returns: Nothing. Saves Pandas dataframe on csv_path specified.

        '''
        if self.results is None:
            raise ValueError('results parameter is None, please run an evaluation first')

        return utils.save_results(self.results, id, csv_path, mode, round_decimals, show_id)

    def get_sensitivity_per_samples(self, csv_path=None, round_decimals=4):
        '''

        Args:
            id: Name of the results evaluation
            csv_path: If specified, results will be saved on that location
            mode: Mode of results. "average" will show the average metrics while "individual" will show metrics by class
            round_decimals: Decimal position to round the numbers.
            show_id: Show id in the first column.

        Returns: Nothing. Saves Pandas dataframe on csv_path specified.

        '''
        if self.results is None:
            raise ValueError('results parameter is None, please run an evaluation first')

        results_classes = self.show_results('individual', round_decimals=round_decimals)

        if self.top_k > 1:
            sensitivity = 'sensitivity_top_1'
        else:
            sensitivity = 'sensitivity'

        results_classes = results_classes[results_classes.columns.intersection(['class', sensitivity, '% of samples'])]
        results_classes = results_classes.sort_values(by=sensitivity).reset_index()
        if csv_path is not None:
            results_classes.to_csv(csv_path)
        return results_classes

    def plot_sensitivity_per_samples(self, csv_path=None, round_decimals=4):
        if self.results is None:
            raise ValueError('results parameter is None, please run an evaluation first')

        results_classes = self.get_sensitivity_per_samples(csv_path, round_decimals)
        visualizer.scatter_plot(results_classes['% of samples'],
                                results_classes['sensitivity_top_1'],
                                '% of Samples',
                                'Sensitivity',
                                'Sensitivity per Samples %')
        return results_classes

    def ensemble_models(self, input_shape, combination_mode='average', ensemble_name='ensemble', model_filename=None):
        ensemble = utils.ensemble_models(self.models, input_shape=input_shape, combination_mode=combination_mode,
                                         ensemble_name=ensemble_name)
        if model_filename is not None:
            ensemble.save(model_filename)
        return ensemble
