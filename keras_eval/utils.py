import os
import json
import scipy
import numpy as np
import keras.models
import tensorflow as tf
import pandas as pd

from keras.layers import average, maximum
from keras.models import Model, Input
from keras.preprocessing import image
from keras_model_specs import ModelSpec
from keras_applications import mobilenet
import keras_model_specs.models.custom_layers as custom_layers


def safe_divide(numerator, denominator):
    if denominator == 0:
        return np.nan
    else:
        return numerator / denominator


def round_list(input_list, decimals=7):
    return [round(x, ndigits=decimals) for x in input_list]


def create_default_custom_objects():
    '''

    Returns: Default custom objects for Keras models supported in keras_applications

    '''
    return {'relu6': mobilenet.relu6, "tf": tf, 'Scale': custom_layers.Scale}


def load_multi_model(models_dir, custom_objects=None):
    '''
    Loads multiple models stored in `models_path`.

    Args:
       models_path: a string indicating the directory were models are stored.
       custom_objects: dict mapping class names (or function names) of custom (non-Keras) objects to class/functions.
                    e.g. for mobilenet models: {'relu6': mobilenet.relu6, 'DepthwiseConv2D': mobilenet.DepthwiseConv2D}

    Returns: List of models, list of model_specs

    '''

    models = []
    model_specs = []
    num_models = 0
    model_extensions = ['.h5', '.hdf5']

    for dirpath, dirnames, files in os.walk(models_dir):
        for dir in dirnames:
            files = os.listdir(os.path.join(dirpath, dir))
            for filename in files:
                if filename.endswith(tuple(model_extensions)):
                    print('Loading model ', filename)
                    model, model_spec = load_model(os.path.join(dirpath, dir, filename), custom_objects=custom_objects)
                    models.append(model)
                    model_specs.append(model_spec)
                    num_models += 1

    print('Models loaded: ', num_models)
    return models, model_specs


def load_model(model_path, specs_path=None, custom_objects=None):
    '''

    Args:
        model_dir: Folder containing the model
        specs_path: If specified custom model_specs name, default `model_spec.json`
        custom_objects: dict mapping class names (or function names) of custom (non-Keras) objects to class/functions.
                    e.g. for mobilenet models: {'relu6': mobilenet.relu6, 'DepthwiseConv2D': mobilenet.DepthwiseConv2D}

    Returns: keras model, model_spec object for that model

    '''
    model = keras.models.load_model(model_path, custom_objects)
    if specs_path is None:
        model_name = model_path.split('/')[-1]
        specs_path = model_path.replace(model_name, 'model_spec.json')
    with open(specs_path) as f:
        model_spec_json = json.load(f)
        model_spec = ModelSpec(model_spec_json)
    return model, model_spec


def create_concepts_default(num_classes):
    concepts_by_default = []
    for i in range(0, num_classes):
        concepts_by_default.append({'id': 'C_' + str(i), 'label': 'Class_' + str(i)})
    return concepts_by_default


def get_concept_items(concepts, key):
    return [concept[key] for concept in concepts]


def create_image_generator(data_dir, batch_size, model_spec):
    '''
    Creates a Keras image generator
    Args:
        batch_size: N images per batch
        preprocessing_function: Function to preprocess the images
        target_size: Size of the images

    Returns: Keras generator without shuffling samples and ground truth labels associated with generator

    '''
    test_gen = image.ImageDataGenerator(preprocessing_function=model_spec.preprocess_input)
    print('Input image size: ', model_spec.target_size)
    generator = test_gen.flow_from_directory(data_dir, batch_size=batch_size, target_size=model_spec.target_size[:2],
                                             class_mode='categorical', shuffle=False)

    labels = keras.utils.np_utils.to_categorical(generator.classes, generator.num_classes)

    return generator, labels


def load_preprocess_image(img_path, model_spec):
    """
    Return a preprocessed image (probably to use within a deep neural net).

    Args:
        img_name: a string indicating the name and path of the image.
        preprocess_func: a preprocessing function to apply to the image.
        target_size: size to resize the image to.

    Returns: the preprocessed image.

    """
    return model_spec.load_image(img_path)


def load_preprocess_images(folder_path, model_spec):
    """
    Return an array of preprocessed images.

    Args:
        img_paths: a list of paths to images.
        preprocess_func: a preprocessing function to apply to each image.
        target_size: size the image should be resized to.

    Returns:
        pre_imgs: an array of preprocessed images.

    """
    images = []
    image_paths = []

    for file_path in sorted(os.listdir(folder_path)):
        if file_path.endswith(".png") or file_path.endswith(".jpeg") or file_path.endswith(".jpg"):
            img_path = os.path.join(folder_path, file_path)
            images.append(load_preprocess_image(img_path, model_spec)[0])
            image_paths.append(img_path)

    return images, image_paths


def combine_probabilities(probabilities, combination_mode='arithmetic'):
    '''
    Args:
        probabilities: Probabilities given by the ensemble of models
        combination_mode: combination_mode: 'arithmetic' / 'geometric' / 'harmonic' mean of the predictions or 'maximum'
           probability value

    Returns: Probabilities combined
    '''

    combiners = {
        'arithmetic': np.mean,
        'geometric': scipy.stats.gmean,
        'harmonic': scipy.stats.hmean,
        'maximum': np.amax
    }

    # Probabilities of the ensemble input=[n_models, n_samples, n_classes] --> output=[n_samples, n_classes]

    # Make sure we have a numpy array
    probabilities = np.array(probabilities)

    # Join probabilities given by an ensemble of models following combination mode
    if probabilities.ndim == 3:
        if probabilities.shape[0] <= 1:
            return probabilities[0]
        else:
            # Combine ensemble probabilities
            if combination_mode not in combiners.keys():
                raise ValueError('Error: invalid option for `combination_mode` ' + str(combination_mode))
            else:
                return combiners[combination_mode](probabilities, axis=0)

    elif probabilities.ndim == 2:
        return probabilities
    else:
        raise ValueError('Incorrect shape for `probabilities` array, we accept [n_samples, n_classes] or '
                         '[n_models, n_samples, n_classes]')


def show_results(results, concepts, id='default_model', mode='average', csv_path=None, round_decimals=3):

    if mode not in ['average', 'individual']:
        raise ValueError('results mode must be either "average" or "individual"')

    if mode is 'average':
        df = pd.DataFrame({'model': id}, index=range(1))

        for metric in results['average'].keys():
            if metric is not 'confusion_matrix':
                if not isinstance(results['average'][metric], list):
                    df[metric] = round(results['average'][metric], round_decimals)
                else:
                    if len(results['average'][metric]) == 1:
                        df[metric] = round(results['average'][metric][0], round_decimals)
                    else:
                        for k in range(len(results['average'][metric])):
                            df[metric + '_top_' + str(k + 1)] = round(results['average'][metric][k], round_decimals)

    if mode is 'individual':
        df = pd.DataFrame()
        metrics = results['individual'][0]['metrics'].keys()
        df['class'] = get_concept_items(concepts, key='id')

        for metric in metrics:
            if not isinstance(results['individual'][0]['metrics'][metric], list):
                concept_list = []
                for idx, concept in enumerate(df['class']):
                    concept_list.append(round(results['individual'][idx]['metrics'][metric], round_decimals))
                df[metric] = concept_list
            elif len(results['individual'][0]['metrics'][metric]) == 1:
                concept_list = []
                for idx, concept in enumerate(df['class']):
                    concept_list = round(results['individual'][idx]['metrics'][metric][0], round_decimals)
                df[metric] = concept_list
            else:
                for k in range(len(results['individual'][0]['metrics'][metric])):
                    concept_list = []
                    for idx, concept in enumerate(df['class']):
                        concept_list.append(
                            round(results['individual'][idx]['metrics'][metric][k], round_decimals))
                    df[metric + '_top_' + str(k + 1)] = concept_list

    if csv_path:
        df.to_csv(csv_path, index=False)

    return df


def ensemble_models(models, input_shape, combination_mode='average', ensemble_name='ensemble'):
    '''

    Args:
        models:  list of keras models
        input_shape: tuple containing input shape in tf format (H, W, C)
        combination_mode: the way probabilities will be joined. We support `average` and `maximum`
        ensemble_name: the name of the model that will be returned

    Returns: a model containing the ensemble of the `models` passed. Same `input_shape` will be used for all of them

    '''
    if not len(input_shape) == 3:
        raise ValueError('Incorrect input shape, it should have 3 dimensions (H, W, C)')
    input_shape = Input(input_shape)
    combination_mode_options = ['average', 'maximum']
    # Collect outputs of models in a list
    count = 0
    models_output = []
    for model in models:
        # Keras needs all the models to be named differently
        model.name = 'model_' + str(count)
        models_output.append(model(input_shape))
        count += 1

    # Computing outputs
    if combination_mode in combination_mode_options:
        if combination_mode == 'average':
            out = average(models_output)
        elif combination_mode == 'maximum':
            out = maximum(models_output)
        # Build model from same input and outputs
        ensemble = Model(inputs=input_shape, outputs=out, name=ensemble_name)
    else:
        raise ValueError('Incorrect combination mode selected, we only allow for `average` or `maximum`')

    return ensemble
