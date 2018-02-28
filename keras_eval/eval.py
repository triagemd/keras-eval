import os
import copy
import numpy as np

import keras_eval.utils as utils
import keras_eval.metrics as metrics
import keras_eval.visualizer as visualizer


class Evaluator(object):

    OPTIONS = {
        'data_dir': {'type': str, 'default': None},
        'class_dictionaries': {'type': list, 'default': None},
        'ensemble_models_dir': {'type': None, 'default': None},
        'model_dir': {'type': None, 'default': None},
        'custom_objects': {'type': None, 'default': None},
        'report_dir': {'type': str, 'default': None},
        'loss_function': {'type': str, 'default': 'categorical_crossentropy'},
        'metrics': {'type': list, 'default': ['accuracy']},
        'batch_size': {'type': int, 'default': 1},
        'verbose': {'type': int, 'default': 0},
    }

    def __init__(self, **options):
        for key, option in self.OPTIONS.items():
            if key not in options and 'default' not in option:
                raise ValueError('missing required option: %s' % (key, ))
            value = options.get(key, copy.copy(option.get('default')))
            if key == 'custom_objects':
                self.custom_objects = utils.create_default_custom_objects()
                if value is not None:
                    self.custom_objects.update(value)
            else:
                setattr(self, key, value)

        extra_options = set(options.keys()) - set(self.OPTIONS.keys())
        if len(extra_options) > 0:
            raise ValueError('unsupported options given: %s' % (', '.join(extra_options), ))

        self.models = []
        self.model_specs = []

        if self.model_dir is not None:
            self.add_model(model_dir=self.model_dir, custom_objects=self.custom_objects)

        if self.ensemble_models_dir is not None:
            self.add_model_ensemble(models_dir=self.ensemble_models_dir, custom_objects=self.custom_objects)

    def add_model(self, model_dir, specs_path=None, custom_objects=None):
        model, model_spec = utils.load_model(model_dir=model_dir, specs_path=specs_path, custom_objects=custom_objects)
        self.models.append(model)
        self.model_specs.append(model_spec)

    def add_model_ensemble(self, models_dir, custom_objects=None):
        models, model_specs = utils.load_multi_model(models_dir=models_dir, custom_objects=custom_objects)
        for i, model in enumerate(models):
            self.models.append(model)
            self.model_specs.append(model_specs[i])

    def remove_model(self, model_index):
        self.models.pop(model_index)
        self.model_specs.pop(model_index)

    def set_class_dictionaries(self, class_dictionaries):
        self.class_dictionaries = class_dictionaries

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

        data_dir = data_dir or self.data_dir

        if data_dir is None:
            raise ValueError('No data directory found, please specify a valid data directory under variable `data_dir`')
        else:
            # Create Keras image generator and obtain predictions
            self.probs, self.labels = self.compute_probabilities_generator(data_dir=data_dir)

            # Create dictionary containing class names
            if self.class_dictionaries is None:
                self.class_dictionaries = utils.create_class_dictionary_default(self.num_classes)

            # Obtain Abbreviations to show on the metrics results
            self.class_abbrevs = utils.get_class_dictionaries_items(self.class_dictionaries, key='abbrev')

            # Check if we have an ensemble or just one model predictions
            if self.probs.shape[0] <= 1:
                self.probs_combined = self.probs[0]
            else:
                # Combine ensemble probabilities
                if combination_mode is not None:
                    self.probs_combined = utils.combine_probs(self.probs, combination_mode)
                else:
                    raise ValueError('You have multiple models, please enter a valid probability `combination_mode`')

            # Compute metrics
            self.get_metrics(self.probs_combined, self.labels, self.class_abbrevs, K, filter_indices)

            # Show metrics visualization as a confusion matrix
            if confusion_matrix:
                self.plot_confusion_matrix(self.probs_combined, self.labels, self.class_abbrevs, save_confusion_matrix_path)

        return self.probs, self.labels

    def plot_confusion_matrix(self, probs, labels, class_names, save_path=None):
        '''

        Args:
            probs: Probabilities from softmax layer
            labels: Ground truth labels
            class_names: List containing the class names
            save_path: If path specified save confusion matrix there

        Returns: Shows the confusion matrix in the screen

        '''

        visualizer.plot_confusion_matrix(probs=probs, labels=labels, class_names=class_names, save_path=save_path)

    def get_metrics(self, probs, labels, class_names, K=(1, 2), list_indices=None, verbose=1):
        '''
        Print to screen metrics from experiment given probs and labels

        Args:
            probs: Probabilities from softmax layer
            labels: Ground truth labels
            K: a tuple of the top-k predictions to consider. E.g. K = (1,2,3,4,5) is top-5 preds
            list_indices: If given take only the predictions corresponding to that indices to compute metrics

        Returns: Dictionary with metrics for each class

        '''
        y_true = labels.argmax(axis=1)

        if list_indices is not None:
            probs = probs[list_indices]
            y_true = y_true[list_indices]

        for k in K:
            acc_k = metrics.accuracy_top_k(probs, y_true, k=k)
            print('Accuracy at k=%i is %.4f' % (k, acc_k))

        # Print sensitivity and precision for different values of K.
        met = metrics.metrics_top_k(probs, y_true, class_names=class_names, k_vals=K, verbose=verbose)
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
            self.num_classes = generator.num_classes
            probs = np.array(probs)
            return probs, labels

    def predict(self, data_dir=None):
        data_dir = data_dir or self.data_dir
        if os.path.isdir(data_dir):
            return self._predict_folder(data_dir)
        elif data_dir.endswith(".png") or data_dir.endswith(".jpeg") or data_dir.endswith(".jpg"):
            return self._predict_image(data_dir)
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
            images, images_path = utils.load_preprocess_images(folder_path, self.model_specs[i])
            images = np.array(images)
            # Predict
            print('Making predictions from model ', str(i))
            probs.append(model.predict(images, batch_size=self.batch_size, verbose=1))

        probs = np.array(probs)

        return probs, images_path

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

        return probs
