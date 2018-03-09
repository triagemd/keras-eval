import os
import copy
import numpy as np
from math import log
import keras_eval.utils as utils
import keras_eval.metrics as metrics
import keras_eval.visualizer as visualizer


class Evaluator(object):

    OPTIONS = {
        'data_dir': {'type': str, 'default': None},
        'class_dictionaries': {'type': list, 'default': None},
        'ensemble_models_dir': {'type': None, 'default': None},
        'model_path': {'type': None, 'default': None},
        'custom_objects': {'type': None, 'default': None},
        'report_dir': {'type': str, 'default': None},
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
            else:
                setattr(self, key, value)

        extra_options = set(options.keys()) - set(self.OPTIONS.keys())
        if len(extra_options) > 0:
            raise ValueError('unsupported options given: %s' % (', '.join(extra_options), ))

        self.models = []
        self.model_specs = []

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

    def set_class_dictionaries(self, class_dictionaries):
        self.class_dictionaries = class_dictionaries

    def get_complete_image_paths(self, filenames):
        image_paths = []
        for filename in filenames:
            image_paths.append(os.path.join(self.data_dir, filename))
        return image_paths

    def evaluate(self, data_dir=None, K=[1], filter_indices=None, confusion_matrix=False,
                 save_confusion_matrix_path=None, combination_mode='arithmetic'):
        '''
        Evaluate a set of images. Each sub-folder under 'data_dir/' will be considered as a different class.
        E.g. 'data_dir/class_1/dog.jpg' , 'data_dir/class_2/cat.jpg

        Args:
            data_dir: Data directory to load the images from
            K: A tuple of the top-k predictions to consider. E.g. K = (1,2,3,4,5) is top-5 preds
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
            self.probs, self.labels = self.compute_probabilities_generator(data_dir=self.data_dir)

            # Create dictionary containing class names
            if self.class_dictionaries is None:
                self.class_dictionaries = utils.create_class_dictionary_default(self.num_classes)

            # Obtain Abbreviations to show on the metrics results
            self.class_abbrevs = utils.get_class_dictionaries_items(self.class_dictionaries, key='abbrev')

            # Compute metrics
            self.get_metrics(probs=self.probs, labels=self.labels, combination_mode=combination_mode,
                             class_names=self.class_abbrevs, K=K, filter_indices=filter_indices,
                             confusion_matrix=confusion_matrix, save_confusion_matrix_path=save_confusion_matrix_path)

        return self.probs, self.labels

    def plot_confusion_matrix(self, probs, labels, class_names=None, save_path=None):
        '''

        Args:
            probs: Probabilities from softmax layer
            labels: Ground truth labels
            class_names: List containing the class names
            save_path: If path specified save confusion matrix there

        Returns: Shows the confusion matrix in the screen

        '''
        class_names = class_names or utils.get_class_dictionaries_items(self.class_dictionaries, key='abbrev')
        visualizer.plot_confusion_matrix(probs=probs, labels=labels, class_names=class_names, save_path=save_path)

    def get_metrics(self, probs, labels, combination_mode=None, K=(1, 2), class_names=None, filter_indices=None,
                    confusion_matrix=False, save_confusion_matrix_path=None, verbose=1):
        '''
        Print to screen metrics from experiment given probs and labels

        Args:
            probs: Probabilities from softmax layer
            labels: Ground truth labels
            K: a tuple of the top-k predictions to consider. E.g. K = (1,2,3,4,5) is top-5 preds
            filter_indices: If given take only the predictions corresponding to that indices to compute metrics

        Returns: Dictionary with metrics for each class

        '''

        # Check if we have an ensemble or just one model predictions
        if len(probs.shape) == 3:
            if probs.shape[0] <= 1:
                self.probs_combined = probs[0]
            else:
                # Combine ensemble probabilities
                if combination_mode is not None:
                    self.probs_combined = utils.combine_probs(probs, combination_mode)
                else:
                    raise ValueError('You have multiple models, please enter a valid probability `combination_mode`')
            probs = self.probs_combined

        class_names = class_names or utils.get_class_dictionaries_items(self.class_dictionaries, key='abbrev')

        if filter_indices is not None:
            probs = probs[filter_indices]
            labels = labels[filter_indices]

        y_true = labels.argmax(axis=1)

        for k in K:
            acc_k = metrics.accuracy_top_k(probs, y_true, k=k)
            print('Accuracy at k=%i is %.4f' % (k, acc_k))

        # Print sensitivity and precision for different values of K.
        met = metrics.metrics_top_k(probs, y_true, class_names=class_names, k_vals=K, verbose=verbose)

        # Show metrics visualization as a confusion matrix
        if confusion_matrix:
            self.plot_confusion_matrix(probs=probs, labels=labels,
                                       class_names=class_names, save_path=save_confusion_matrix_path)

        return met

    def compute_probabilities_generator(self, data_dir=None):
        '''

        Args:
            data_dir: Data directory to load the images from

        Returns: Probabilities, ground truth labels of predictions

        '''
        probs = []
        if len(self.models) < 1:
            raise ValueError('No models found, please add a valid Keras model first')
        else:
            for i, model in enumerate(self.models):
                print('Making predictions from model ', str(i))
                generator, labels = utils.create_image_generator(data_dir, self.batch_size, self.model_specs[i])
                # N_batches + 1 to gather all the images + collect without repetition [0:n_samples]
                probs.append(model.predict_generator(generator=generator,
                                                     steps=(generator.samples // self.batch_size) + 1,
                                                     workers=1,
                                                     verbose=1)[0:generator.samples])

            self.generator = generator
            self.num_classes = generator.num_classes
            self.image_paths = self.get_complete_image_paths(generator.filenames)

            probs = np.array(probs)

            return probs, labels

    def predict(self, data_dir=None):
        '''

        Args:
            data_dir: If folder run _predict_folder, if single image run _predict_image()

        Returns: Probabilities of the folder of images/single image

        '''
        self.data_dir = data_dir or self.data_dir

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
        probs = []
        for i, model in enumerate(self.models):
            # Read images from folder
            images, image_paths = utils.load_preprocess_images(folder_path, self.model_specs[i])
            images = np.array(images)
            # Predict
            print('Making predictions from model ', str(i))
            probs.append(model.predict(images, batch_size=self.batch_size, verbose=1))

        probs = np.array(probs)
        self.image_paths = image_paths

        return probs

    def _predict_image(self, image_path):
        '''

        Predict class probabilities for a single image.

        Args:
            image_path: Path where the image is located

        Returns:

        '''
        probs = []
        for i, model in enumerate(self.models):
            # Read image
            image = utils.load_preprocess_image(image_path, self.model_specs[i])
            # Predict
            print('Making predictions from model ', str(i))
            probs.append(model.predict(image, batch_size=1, verbose=1))

        probs = np.array(probs)
        self.image_paths = [image_path]

        return probs

    def show_threshold_impact(self, probs, labels, type='probability', threshold=None):
        '''
        Interactive Plot showing the effect of the threshold
        Args:
            probs: probabilities given by the model [n_samples,n_classes]
            labels: ground truth labels (categorical)
            type: 'probability' or 'entropy' for a threshold on network top-1 prob or uncertainty in all predictions
            threshold: Custom threshold

        Returns: The index of the images with error or correct per every threshold, and arrays with the percentage.

        '''
        # Get Error Indices, Number of Correct Predictions, Number of Error Predictions per Threshold
        if type == 'probability':
            threshold = threshold or np.arange(0, 1.01, 0.01)
            errors_ind, correct_ind, correct, errors = metrics.get_top1_probability_stats(probs, labels,
                                                                                          threshold, verbose=0)
        elif type == 'entropy':
            threshold = threshold or np.arange(0, log(probs.shape[1], 2), 0.01)
            errors_ind, correct_ind, correct, errors = metrics.get_top1_entropy_stats(probs, labels,
                                                                                      threshold, verbose=0)

        # Uncomment for showing percentage (min threshold have to be 0)
        n_total_errors = errors[0]
        n_total_correct = correct[0]
        errors = ((n_total_errors - errors) / n_total_errors) * 100
        correct = ((correct) / n_total_correct) * 100

        visualizer.plotly_threshold(threshold, correct, errors, title='Top-1 Probability Threshold Tuning')

        return errors_ind, correct_ind, correct, errors

    def get_image_paths_by_prediction(self, probs, labels=None, class_names=None, image_paths=None):
        '''

        Args:
            probs: probabilities given by the model [n_samples,n_classes]
            labels: ground truth labels (categorical) (by default last evaluation)
            class_names: list with class names (by default last evaluation)
            image_paths: list with image_paths (by default last evaluation)

        Returns: A dictionary containing a list of images per confusion matrix square (relation ClassA_ClassB)

        '''
        labels = labels or self.labels
        image_paths = image_paths or self.image_paths
        assert probs.shape[0] == len(image_paths)

        class_names = class_names or utils.get_class_dictionaries_items(self.class_dictionaries, key='abbrev')

        predictions = np.argmax(probs, axis=1)
        y_true = labels.argmax(axis=1)
        dict_image_paths_class = {}

        for name_1 in class_names:
            for name_2 in class_names:
                dict_image_paths_class.update({name_1 + '_' + name_2: []})

        for i, pred in enumerate(predictions):
            predicted_label = class_names[pred]
            correct_label = class_names[y_true[i]]
            list_image_paths = dict_image_paths_class[str(correct_label + '_' + predicted_label)]
            list_image_paths.append(image_paths[i])
            dict_image_paths_class.update({correct_label + '_' + predicted_label: list_image_paths})

        return dict_image_paths_class

    def plot_images(self, image_paths, n_imgs=None, title=''):
        '''

        Args:
            image_paths: list with image_paths
            n_imgs: number of images to show
            title: title for the plot

        Returns:

        '''
        image_paths = np.array(image_paths)
        if n_imgs is None:
            n_imgs = image_paths.shape[0]

        visualizer.plot_images(image_paths, n_imgs, title)
