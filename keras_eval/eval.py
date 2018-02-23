import os
import json
import keras

import keras_eval.utils as utils
import keras_eval.metrics as metrics
import keras_eval.visualizer as visualizer
from keras.preprocessing import image


class Evaluator(object):

    OPTIONS = {
        'target_size': {'type': None},
        'data_dir': {'type': str},
        'class_dictionary': {'type': dict, 'default': None},
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

    def evaluate(self, data_dir=None, target_size=None, K=[1], filter_indices=None, confusion_matrix=False,
                 save_confusion_matrix_path=None):
        data_dir = data_dir or self.data_dir
        if data_dir is None:
            raise ValueError('Please specify a valid data directory under variable data_dir')
        else:
            self.probs, self.labels = self.compute_probabilities_generator(data_dir=None)
            self.print_results(self.probs, self.labels, K, filter_indices)
            if confusion_matrix:
                self.plot_confusion_matrix(self.probs.argmax(axis=1), self.labels.argmax(axis=1),
                                           class_names, save_confusion_matrix_path)
        return self.probs, self.labels

    def plot_confusion_matrix(self, probs, labels, class_names, save_path):
        if probs is None or labels is None:
            probs, labels = self.probs, self.labels

        visualizer.confusion_matrix(probs.argmax(axis=1), labels.argmax(axis=1),
                                    labels=class_names, save_path=save_path)

    def print_results(self, probs, labels, K=(1, 2), list_indices=None):
        '''
        Print to screen metrics from experiment given probs and y_true

        Args:
            probs: Probabilities from softmax layer
            labels: Ground truth
            K: a tuple of the top-k predictions to consider. E.g. K = (1,2,3,4,5) is top-5 preds
            list_indices: If given take only the predictions corresponding to that indices to compute metrics

        Returns: Nothing. Prints results to screen.

        '''
        y_true = labels.argmax(axis=1)

        if list_indices is not None:
            probs = probs[list_indices]
            y_true = y_true[list_indices]

        for k in K:
            acc_k = metrics.accuracy_top_k(y_true, probs, k=k)
            print('Accuracy at k=%i is %.4f' % (k, acc_k))

        # Print sensitivity and precision for different values of K.
        metrics.metrics_top_k(y_true, probs, labels=self.skindata.label_abbrevs(), k_vals=K, print_screen=True)

    def compute_probabilities_generator(self, data_dir=None):
        probs = []
        for i, model in enumerate(self.models):
            if len(target_size) == len(models):
                target_size_images = target_size[i]
            else:
                target_size_images = target_size

            generator, labels = utils.create_image_generator(data_dir,
                                                             self.batch_size,
                                                             self.model_specs[i]['target_size'],
                                                             self.model_specs[i]['preprocess_func'])
            print('Making predictions from model ', str(i))
            probs.append(model.predict_generator(generator=generator,
                                                 steps=(generator.samples // batch_size) + 1,
                                                 workers=1,
                                                 verbose=1))
        probs = np.array(probs)
        return probs, labels

    def predict(self, folder_path, target_size=None):
        target_size = target_size or self.target_size
        probs = []
        for i, model in enumerate(self.models):
            if len(target_size) == len(self.models):
                target_size_images = target_size[i]
            else:
                target_size_images = target_size

            images = []
            images_path = []

            # Read images from folder
            for file in sorted(os.listdir(folder_path)):
                if file.endswith(".png") or file.endswith(".jpeg") or file.endswith(".jpg"):
                    images.append(load_preprocess_image(
                        os.path.join(folder_path, file), preprocess_func=self.preprocess_input,
                        target_size=target_size_images)[0])
                    images_path.append(os.path.join(folder_path, file))
            images = np.array(images)

            # Predict
            print('Making predictions from model ', str(i))
            probs.append(model.predict(images, batch_size=batch_size, verbose=1))

        probs = np.array(probs)

        return probs, images_path
