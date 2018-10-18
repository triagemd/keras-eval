import os
import json
import scipy.stats
import numpy as np
import keras.models
import tensorflow as tf
import pandas as pd

from copy import deepcopy
from keras.layers import average, maximum
from keras.models import Model, Input
from keras.preprocessing import image
from keras_model_specs import ModelSpec
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
    return {'tf': tf, 'Scale': custom_layers.Scale}


def load_multi_model(models_dir, custom_objects=None):
    '''
    Loads multiple models stored in `models_path`.

    Args:
       models_path: A string indicating the directory were models are stored.
       custom_objects: Dict mapping class names (or function names) of custom (non-Keras) objects to class/functions.

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
        custom_objects: Dict mapping class names (or function names) of custom (non-Keras) objects to class/functions.

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


def ensemble_models(models, input_shape, combination_mode='average', ensemble_name='ensemble'):
    '''

    Args:
        models: List of keras models
        input_shape: Tuple containing input shape in tf format (H, W, C)
        combination_mode: The way probabilities will be joined. We support `average` and `maximum`
        ensemble_name: The name of the model that will be returned

    Returns: A model containing the ensemble of the `models` passed. Same `input_shape` will be used for all of them

    '''
    if not len(input_shape) == 3:
        raise ValueError('Incorrect input shape, it should have 3 dimensions (H, W, C)')
    input_shape = Input(input_shape)
    combination_mode_options = ['average', 'maximum']
    # Collect outputs of models in a list

    models_output = []
    for i, model in enumerate(models):
        # Keras needs all the models to be named differently
        model.name = 'model_' + str(i)
        models_output.append(model(input_shape))

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


def get_default_concepts(data_dir):
    '''
    Creates default concepts dictionary from data_dir folder names
    Args:
        data_dir: String indicating the path where the concept folders are

    Returns:
        concepts: Dictionary with 'label' and 'id' equal to each folder name
    '''

    if not os.path.exists(data_dir):
        raise ValueError('data_dir path does not exist')

    concepts = []
    for directory in sorted(os.listdir(data_dir)):
        if os.path.isdir(os.path.join(data_dir, directory)):
            concepts.append({'label': directory, 'id': directory})
    return concepts


def get_concept_items(concepts, key):
    return [concept[key] for concept in concepts]


def read_dictionary(dictionary_path):
    if os.path.exists(dictionary_path):
        with open(dictionary_path, 'r') as dictionary_file:
            dictionary = json.load(dictionary_file)
    else:
        raise ValueError('Error: invalid dictionary path' + str(dictionary_path))
    return dictionary


def create_training_json(train_dir, output_json_file):
    '''
    Checks if evaluation concepts are unique
    Args:
        train_dir: The location where you have the training directory
        output_json_file: The output file name and path e.g.: ./dictionary.json

    Returns:
        True, if there are no repeat concepts, else raises error
    '''
    concept_dict = []
    train_concepts = get_default_concepts(train_dir)
    for idx in range(len(train_concepts)):
        concept_dict.append({"class_index": idx, "class_name": train_concepts[idx]["label"], "group": train_concepts[idx]["label"]})
    with open(output_json_file, 'w') as file_obj:
        json.dump(concept_dict, file_obj, indent=4, sort_keys=True)


def check_concept_unique(concept_dict):
    '''
    Checks if evaluation concepts are unique
    Args:
        concept_dict: Dictionary that contains class_id, train_concepts and groups
    Returns:
        True, if there are no repeat concepts, else raises error
    '''
    concept_class_name_dict = {}
    for concept_dict_item in concept_dict:
        if concept_dict_item['class_name'] in concept_class_name_dict:
            raise ValueError("Concept has been repeated:", concept_dict_item['class_name'])
        else:
            concept_class_name_dict[concept_dict_item['class_name']] = 1

    return True


def compare_group_test_concepts(test_concepts_list, concept_dict):
    '''
    Checks if concept dictionary has the groups as the test concepts
    Args:
        test_concepts_list: List of labels corresponding to the test concepts
        concept_dict: Dictionary that contains class_id, train_concepts and groups
    Returns:
        True, if there are no repeat concepts, else raises error
    '''
    concept_group_list = get_concept_items(concept_dict, key="group")

    different_concept_set = set(concept_group_list).symmetric_difference(set(test_concepts_list))
    if len(different_concept_set):
        raise ValueError(
            "The following concepts are not present in either the concept dictionary or among the test classes:",
            list(different_concept_set))

    else:
        return True


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
        img_name: A string indicating the name and path of the image.
        preprocess_func: A preprocessing function to apply to the image.
        target_size: Size to resize the image to.

    Returns: The preprocessed image.

    """
    return model_spec.load_image(img_path)


def load_preprocess_images(folder_path, model_spec):
    """
    Return an array of preprocessed images.

    Args:
        img_paths: A list of paths to images.
        preprocess_func: A preprocessing function to apply to each image.
        target_size: Size the image should be resized to.

    Returns:
        pre_imgs: An array of preprocessed images.

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
        combination_mode: Combination_mode: 'arithmetic' / 'geometric' / 'harmonic' mean of the predictions or 'maximum'
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


def show_results(results, id='default_model', mode='average', csv_path=None, round_decimals=3):
    '''

    Converts results to pandas to show a nice visualization of the results. Allow saving them to a csv file.

    Args:
        results: Results dictionary provided by the evaluation (evaluator.results)
        id: Name of the results evaluation
        mode: Mode of results. "average" will show the average metrics while "individual" will show metrics by class
        csv_path: If specified, results will be saved on that location
        round_decimals: Position to round the numbers.

    Returns: A pandas dataframe with the results and prints a nice visualization

    '''

    if mode not in ['average', 'individual']:
        raise ValueError('Results mode must be either "average" or "individual"')

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
        df['class'] = [result['concept'] for result in results['individual']]

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


def compute_differential_results(results_1, results_2):
    '''

    Given two results dictionaries this function will compute the difference between both

    Args:
        results_1: Array of results (1)
        results_2: Array of results (2)

    Returns: The resulting differential results dictionary

    '''

    if len(results_1['average']) != len(results_2['average']):
        raise ValueError('Results length do not match for "average" values')

    if len(results_1['individual']) != len(results_2['individual']):
        raise ValueError('Results length do not match for "individual" values')

    # new array to store results
    differential_results = deepcopy(results_1)

    for metric in differential_results['average'].keys():
        if metric not in ['number_of_samples', 'number_of_classes']:
            if not isinstance(results_1['average'][metric], list):
                differential_results['average'][metric] = results_1['average'][metric] - \
                                                          results_2['average'][metric]
            else:
                if len(results_1['average'][metric]) == 1:
                    differential_results['average'][metric] = results_1['average'][metric][0] - \
                                                              results_2['average'][metric][0]
                else:
                    for k in range(len(results_1['average'][metric])):
                        differential_results['average'][metric][k] = \
                            results_1['average'][metric][k] - results_2['average'][metric][k]

    for index, category in enumerate(differential_results['individual']):
        for metric in category['metrics'].keys():
            if not isinstance(category['metrics'][metric], list):
                differential_results['individual'][index]['metrics'][metric] = \
                    results_1['individual'][index]['metrics'][metric] - \
                    results_2['individual'][index]['metrics'][metric]
            else:
                if len(results_1['individual'][metric]) == 1:
                    differential_results['individual'][index]['metrics'][metric] = \
                        results_1['individual'][index]['metrics'][metric][0] - \
                        results_2['individual'][index]['metrics'][metric][0]
                else:
                    for k in range(len(results_1['individual'][metric])):
                        differential_results['individual'][index]['metrics'][metric][k] = \
                            results_1['individual'][index]['metrics'][metric][k] - \
                            results_2['individual'][index]['metrics'][metric][k]

    return differential_results
