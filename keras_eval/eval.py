import os
import json
import keras

import keras_eval.utils as utils
from keras.preprocessing import image


class Evaluator(object):

    OPTIONS = {
        'target_size': {'type': None},
        'data_dir': {'type': str},
        'model_spec': {'type': str},
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

        if self.model_dir is not None:
            self.models = []
            self.models.append(utils.load_model(self.model_dir))

        if self.ensemble_models_dir is not None:
            self.models = []
            self.models.append(utils.load_multi_model(self.model_dir))

    def add_model(self, model_path):
        self.models.append(utils.load_model(model_path))

    def remove_model(self, model_index):
        self.models.pop(model_index)

    def evaluate(self, data_dir=None, target_size=None):
        target_size = target_size or self.target_size
        data_dir = data_dir or self.data_dir

        if self.data_dir is not None:
            for i, model in enumerate(models):
                if len(target_size) == len(models):
                    target_size_images = target_size[i]
                else:
                    target_size_images = target_size

                generator, labels = utils.create_image_generator(data_dir, batch_size, target_size_images)
                model.predict_generator(generator=generator,
                                        steps=(generator.samples // batch_size) + 1,
                                        workers=1,
                                        verbose=1)
        else:
            raise ValueError('Please specify a data directory under variable data_dir')

    def predict(self, folder_path, target_size=None):
        target_size = target_size or self.target_size

        for i, model in enumerate(models):
            if len(target_size) == len(models):
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
            print('Making predictions from model ', str(model_index))
            probs.append(model.predict(images, batch_size=batch_size, verbose=1))

        probs = np.array(probs)

        return probs, images_path
