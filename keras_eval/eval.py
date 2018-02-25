import os
import json
import keras
import copy

import keras_eval.utils as utils
import keras_eval.metrics as metrics
import keras_eval.visualizer as visualizer
from keras.preprocessing import image
from keras_model_specs import ModelSpec


class Evaluator(object):

    OPTIONS = {
        'data_dir': {'type': str, 'default': None},
        'class_dictionaries': {'type': list, 'default': None},
        'preprocess_input': {'type': None, 'default': None},
        'ensemble_models_dir': {'type': None, 'default': None},
        'model_dir': {'type': None, 'default': None},
        'report_dir': {'type': str, 'default': None},
        'data_generator': {'type': None, 'default': None},
        'loss_function': {'type': str, 'default': 'categorical_crossentropy'},
        'metrics': {'type': list, 'default': ['accuracy']},
        'batch_size': {'type': int, 'default': 1},
        'num_gpus': {'type': int, 'default': 0},
        'verbose': {'type': str, 'default': False},
    }

    def __init__(self, **options):
        for key, option in self.OPTIONS.items():
            if key not in options and 'default' not in option:
                raise ValueError('missing required option: %s' % (key, ))
            value = options.get(key, copy.copy(option.get('default')))
            setattr(self, key, value)

        extra_options = set(options.keys()) - set(self.OPTIONS.keys())
        if len(extra_options) > 0:
            raise ValueError('unsupported options given: %s' % (', '.join(extra_options), ))

        self.models = []
        self.model_specs = []

        if self.model_dir is not None:
            self.models.append(utils.load_model(self.model_dir))

        if self.ensemble_models_dir is not None:
            self.models.append(utils.load_multi_model(self.model_dir))

    def add_model(self, model_path):
        model, model_spec = utils.load_model(model_path)
        self.models.append(model)
        self.model_specs.append(model_spec)

    def remove_model(self, model_index):
        self.models.pop(model_index)
        self.model_specs.pop(model_index)

    def evaluate(self, data_dir=None, K=[1], filter_indices=None, confusion_matrix=False,
                 save_confusion_matrix_path=None):
        data_dir = data_dir or self.data_dir
        if data_dir is None:
            raise ValueError('Please specify a valid data directory under variable data_dir')
        else:
            self.probs, self.labels = self.compute_probabilities_generator(data_dir=data_dir)

            if self.class_dictionaries is None:
                self.class_dictionaries = utils.class_dictionary_default(self.num_classes)

            self.class_abbrevs = utils.get_class_dictionaries_items(class_dictionaries, key='abbrev')

            self.get_metrics(self.probs, self.labels, self.class_abbrevs, K, filter_indices)
            if confusion_matrix:
                self.plot_confusion_matrix(self.probs, self.labels, self.class_abbrevs, save_confusion_matrix_path)
        return self.probs, self.labels

    def plot_confusion_matrix(self, probs, labels, class_names, save_path):
        if probs is None or labels is None:
            probs, labels = self.probs, self.labels

        visualizer.confusion_matrix(probs, labels, class_names, save_path=save_path)

    def get_metrics(self, probs, labels, class_names, K=(1, 2), list_indices=None, verbose=1):
        '''
        Print to screen metrics from experiment given probs and labels

        Args:
            probs: Probabilities from softmax layer
            labels: Ground truth labels
            K: a tuple of the top-k predictions to consider. E.g. K = (1,2,3,4,5) is top-5 preds
            list_indices: If given take only the predictions corresponding to that indices to compute metrics

        Returns: Dictionary with metrics

        '''
        y_true = labels.argmax(axis=1)

        if list_indices is not None:
            probs = probs[list_indices]
            y_true = y_true[list_indices]

        for k in K:
            acc_k = metrics.accuracy_top_k(y_true, probs, k=k)
            print('Accuracy at k=%i is %.4f' % (k, acc_k))

        # Print sensitivity and precision for different values of K.
        met = metrics.metrics_top_k(probs, y_true, class_names=class_names, k_vals=K, verbose=verbose)
        return met

    def compute_probabilities_generator(self, data_dir=None):
        probs = []
        for i, model in enumerate(self.models):
            generator, labels = utils.create_image_generator(data_dir, self.batch_size, self.model_specs[i])
            print('Making predictions from model ', str(i))
            probs.append(model.predict_generator(generator=generator, steps=(generator.samples // batch_size) + 1,
                                                 workers=1,
                                                 verbose=1))
        self.num_classes = generator.num_classes
        probs = np.array(probs)
        return probs, labels

    def predict(self, folder_path):
        probs = []
        for i, model in enumerate(self.models):
            # Read images from folder
            file_paths = sorted(os.listdir(folder_path))
            images, images_path = utils.load_preprocess_images(file_paths, model_specs[i])
            images = np.array(images)

            # Predict
            print('Making predictions from model ', str(i))
            probs.append(model.predict(images, batch_size=self.batch_size, verbose=1))

        probs = np.array(probs)

        return probs, images_path

    def predict_image(self, image_path):
        probs = []
        for i, model in enumerate(self.models):
            # Read image
            image = utils.load_preprocess_image(image_path, model_specs[i])
            # Predict
            print('Making predictions from model ', str(i))
            probs.append(model.predict(image, batch_size=1, verbose=1))

        probs = np.array(probs)

        return probs
