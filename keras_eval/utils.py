import os
import json
import scipy.stats
import numpy as np
import keras.models
import tensorflow as tf
import pandas as pd
import keras_model_specs.models.custom_layers as custom_layers

from keras.layers import average, maximum
from keras.models import Model, Input
from keras.preprocessing import image
from keras_model_specs import ModelSpec
from keras_eval.data_generators import AugmentedImageDataGenerator


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


def get_dictionary_concepts(model_dictionary):
    '''
    Returns concept list from a model dictionary.

    Args:
        model_dictionary: String indicating the path where the model_dictionary json file is.
                        This dictionary must contain 'class_index' and 'class_name' for each class.

    Returns:
        concepts: Dictionary with 'label' and 'id' equal to 'class_name' and 'class_index' for each class.
    '''
    if not os.path.isfile(model_dictionary):
        raise ValueError('model_dictionary file does not exist')

    concepts = []
    for model_class in read_dictionary(model_dictionary):
        concepts.append({'label': model_class['class_name'], 'id': model_class['class_index']})
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


def check_input_samples(y_probs, y_true):
    '''
    Checks if number predicted samples from 'y_probs' is the same as the ground truth samples from 'y_true'
    Args:
        y_probs: A numpy array of the class probabilities.
        y_true: A numpy array of the true class labels (*not* encoded as 1-hot).
    Returns:
        True, if len(y_probs) == len(y_true), otherwise raises error
    '''
    if len(y_probs) != len(y_true):
        raise ValueError('The number predicted samples (%i) is different from the ground truth samples (%i)' %
                         (len(y_probs), len(y_true)))
    else:
        return True


def check_top_k_concepts(concepts, top_k):
    '''
    Checks if the 'top_k' requested is not higher than the number of 'concepts', or zero.
    Args:
        concepts: A list containing the names of the classes.
        top_k: A number specifying the top-k results to compute. E.g. 2 will compute top-1 and top-2
    Returns:
        True, if len(top_k)>0 && len(top_k)>len(concepts), otherwise raises error
    '''
    if top_k <= 0 or top_k > len(concepts):
        raise ValueError('`top_k` value should be between 1 and the total number of concepts (%i)' % len(concepts))
    else:
        return True


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


def create_image_generator(data_dir, batch_size, model_spec, data_augmentation=None):
    '''
    Creates a Keras Image Generator.
    Args:
        batch_size: N images per batch
        preprocessing_function: Function to preprocess the images
        target_size: Size of the images

    Returns: Keras generator without shuffling samples and ground truth labels associated with generator

    '''
    if data_augmentation is None:
        test_gen = image.ImageDataGenerator(preprocessing_function=model_spec.preprocess_input)
        generator = test_gen.flow_from_directory(data_dir, batch_size=batch_size,
                                                 target_size=model_spec.target_size[:2],
                                                 class_mode='categorical', shuffle=False)
    else:
        test_gen = AugmentedImageDataGenerator(preprocessing_function=model_spec.preprocess_input,
                                               data_augmentation=data_augmentation)
        generator = test_gen.flow_from_directory(data_dir, batch_size=1,
                                                 target_size=model_spec.target_size[:2],
                                                 class_mode='categorical', shuffle=False)

    print('Input image size: ', model_spec.target_size)

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
    if img_path.endswith(".png") or img_path.endswith(".jpeg") or img_path.endswith(".jpg"):
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


def results_to_dataframe(results, id='default_model', mode='average', round_decimals=3, show_id=True):
    '''

    Converts results to pandas to show a nice visualization of the results. Allow saving them to a csv file.

    Args:
        results: Results dictionary provided by the evaluation (evaluator.results)
        id: Name of the results evaluation
        mode: Mode of results. "average" will show the average metrics while "individual" will show metrics by class
        csv_path: If specified, results will be saved on that location
        round_decimals: Decimal position to round the numbers.
        show_id: Show id in the first column.

    Returns: A pandas dataframe with the results and prints a nice visualization

    '''

    if mode not in ['average', 'individual']:
        raise ValueError('Results mode must be either "average" or "individual"')

    if mode == 'average':
        df = pd.DataFrame({'id': id}, index=range(1))

        for metric in results['average'].keys():
            if metric != 'confusion_matrix':
                if not isinstance(results['average'][metric], list):
                    df[metric] = round(results['average'][metric], round_decimals)
                else:
                    if len(results['average'][metric]) == 1:
                        df[metric] = round(results['average'][metric][0], round_decimals)
                    else:
                        for k in range(len(results['average'][metric])):
                            df[metric + '_top_' + str(k + 1)] = round(results['average'][metric][k], round_decimals)

    if mode == 'individual':
        df = pd.DataFrame()
        metrics = results['individual'][0]['metrics'].keys()
        df['id'] = [id for i in range(len(results['individual']))]
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
    if not show_id:
        df.drop('id', axis=1, inplace=True)

    return df


def mkdir(path):
    '''

    Args:
        path: Path where directory will be created

    Returns: Nothing. Creates directory with the path specified

    '''
    if not os.path.exists(path):
        os.makedirs(path)


def save_numpy(id, path, file):
    np.save(os.path.join(path, id + '.npy'), file)


def save_results(results, id, csv_path, mode='average', round_decimals=3, show_id=True):
    '''

    Args:
        results: Results dictionary provided by the evaluation (evaluator.results)
        id: Name of the results evaluation
        mode: Mode of results. "average" will show the average metrics while "individual" will show metrics by class
        csv_path: If specified, results will be saved on that location
        round_decimals: Decimal position to round the numbers.
        show_id: Show id in the first column.

    Returns: Nothing. Saves pandas dataframe on csv_path specified.

    '''
    df = results_to_dataframe(results, id=id, mode=mode, round_decimals=round_decimals, show_id=show_id)
    mkdir(csv_path)
    df.to_csv(os.path.join(csv_path, id + '_' + mode + '.csv'), float_format='%.' + str(round_decimals) + 'f',
              index=False)


def load_csv_to_dataframe(csv_paths):
    '''

    Args:
        csv_paths: Path or list of paths to the csvs

    Returns: A Pandas dataframe containing the csv information

    '''
    results_dataframe = []
    if isinstance(csv_paths, list):
        for path in csv_paths:
            results_dataframe.append(pd.read_csv(path))
    elif isinstance(csv_paths, str):
        results_dataframe = pd.read_csv(path)
    else:
        raise ValueError('Incorrect format for `csv_paths`, a list of strings or a single string are expected')
    return results_dataframe


def compute_differential_str(value_reference, value, round_decimals):
    '''

    Args:
        value_reference: Reference Value
        value: Value to modify
        round_decimals: Decimal position to round the numbers.

    Returns: A string with the differential between the two values (value - value_reference)

    '''
    diff_value = round(value - value_reference, round_decimals)
    if diff_value > 0:
        return ' (+' + str(diff_value) + ')'
    else:
        return ' (' + str(diff_value) + ')'


def results_differential(dataframes, mode='average', round_decimals=4, save_csv_path=None):
    '''

    Args:
        dataframes: List of results dataframes. The first one will be considered the reference.
        mode: Mode of results. "average" will show the average metrics while "individual" will show metrics by class
        round_decimals: Decimal position to round the numbers.
        save_csv_path: Path to save the resulting dataframe with the differential information.

    Returns: Modified dataframe with the differential information.

    '''
    if len(dataframes) < 2:
        raise ValueError('The number of dataframes should be higher than 1')

    if mode == 'average':
        skip_values = ['id', 'number_of_samples', 'number_of_classes']
        reference_dataframe = dataframes.pop(0)
        differential_dataframe = reference_dataframe.copy()
        for dataframe in dataframes:
            for name, values in dataframe.iteritems():
                if name not in skip_values:
                    diff_str = compute_differential_str(reference_dataframe[name][0], dataframe[name][0],
                                                        round_decimals)
                    dataframe[name] = str(dataframe[name][0]) + diff_str
            differential_dataframe = pd.concat((differential_dataframe, dataframe), ignore_index=True)

    elif mode == 'individual':
        skip_values = ['id', '% of samples', 'class']
        n_evaluations = len(dataframes)
        differential_dataframe = pd.concat(dataframes, ignore_index=True)
        differential_dataframe = differential_dataframe.rename_axis('index').sort_values(by=['class', 'index'],
                                                                                         ascending=[True, True])
        differential_dataframe = differential_dataframe.reset_index(drop=True)
        reference_index = 0
        for index, row in differential_dataframe.iterrows():
            if index % n_evaluations == 0:
                reference_index = index
            else:
                reference_row = differential_dataframe.iloc[reference_index]
                for name in list(differential_dataframe.columns.values):
                    if name not in skip_values:
                        diff_str = compute_differential_str(reference_row[name], row[name], round_decimals)
                        row[name] = str(round(row[name], round_decimals)) + diff_str
                differential_dataframe.iloc[index] = row

    else:
        raise ValueError('Results mode must be either "average" or "individual"')

    if save_csv_path is not None:
        differential_dataframe.to_csv(save_csv_path, float_format='%.' + str(round_decimals) + 'f', index=False)

    return differential_dataframe


def check_result_type(result_csv_file, individual):
    '''
    Checks if the evaluation results file type is of the required format i.e. individual or average metrics
    Args:
        result_csv_file: csv file name
        individual: Boolean set to True if 'result_csv_file' is individual. Otherwise, set to False.
    Returns: True if the file 'result_csv_file' is of the required type, else False
    '''
    csv_type = result_csv_file[result_csv_file.rfind('_') + 1:-4]
    if individual and csv_type == 'individual' or not individual and csv_type == 'average':
        return True
    elif individual and csv_type == 'average' or not individual and csv_type == 'individual':
        return False
    else:
        raise ValueError('File name not in required format')
