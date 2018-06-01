import os
import pytest
import numpy as np
import tensorflow as tf
import keras_eval.utils as utils

from keras.applications import mobilenet
from keras_model_specs import ModelSpec


@pytest.fixture('session')
def test_dataset_path():
    return os.path.abspath(os.path.join('tests', 'files', 'test', 'catdog'))


@pytest.fixture('session')
def test_folder_image_path():
    return os.path.abspath(os.path.join('tests', 'files', 'catdog', 'test', 'cat'))


@pytest.fixture('session')
def test_image_path():
    return os.path.abspath(os.path.join('tests', 'files', 'catdog', 'test', 'cat', 'cat-1.jpg'))


@pytest.fixture('session')
def model_spec_mobilenet():
    dataset_mean = [142.69182214, 119.05833338, 106.89884415]
    return ModelSpec.get('mobilenet_v1', preprocess_func='mean_subtraction', preprocess_args=dataset_mean)


def test_safe_divide():
    assert np.isnan(utils.safe_divide(10.0, 0.0))
    assert utils.safe_divide(10.0, 5.0) == 2.0


def test_load_model():
    custom_objects = {'relu6': mobilenet.relu6, 'DepthwiseConv2D': mobilenet.DepthwiseConv2D, "tf": tf}
    model_path = 'tmp/fixtures/models/mobilenet_1/mobilenet_v1.h5'
    model_spec_path = 'tmp/fixtures/models/mobilenet_2/model_spec.json'

    # Default model_spec
    model = utils.load_model(model_path, custom_objects=custom_objects)
    assert model

    # Custom model_spec
    model = utils.load_model(model_path, specs_path=model_spec_path, custom_objects=custom_objects)
    assert model


def test_load_model_ensemble():
    ensemble_dir = 'tmp/fixtures/models'
    custom_objects = {'relu6': mobilenet.relu6, 'DepthwiseConv2D': mobilenet.DepthwiseConv2D, "tf": tf}
    models = utils.load_multi_model(ensemble_dir, custom_objects=custom_objects)
    assert models


def test_combine_probabilities():
    # Ensemble 3 models
    probabilities = [[[0.4, 0.6], [0.8, 0.2]], [[0.1, 0.9], [0.2, 0.6]], [[0.4, 0.6], [0.8, 0.2]]]

    # Maximum
    combined_probabilities = utils.combine_probabilities(probabilities, 'maximum')
    assert len(combined_probabilities.shape) == 2
    combined_probabilities_expected = [[0.4, 0.9], [0.8, 0.6]]
    np.testing.assert_array_equal(np.round(combined_probabilities, decimals=2), combined_probabilities_expected)

    # Arithmetic
    combined_probabilities = utils.combine_probabilities(probabilities, 'arithmetic')
    assert len(combined_probabilities.shape) == 2
    combined_probabilities_expected = [[0.3, 0.7], [0.6, 0.33]]
    np.testing.assert_array_equal(np.round(combined_probabilities, decimals=2), combined_probabilities_expected)

    # Geometric
    combined_probabilities = utils.combine_probabilities(probabilities, 'geometric')
    assert len(combined_probabilities.shape) == 2
    combined_probabilities_expected = [[0.25, 0.69], [0.5, 0.29]]
    np.testing.assert_array_equal(np.round(combined_probabilities, decimals=2), combined_probabilities_expected)

    # Harmonic
    combined_probabilities = utils.combine_probabilities(probabilities, 'harmonic')
    assert len(combined_probabilities.shape) == 2
    combined_probabilities_expected = [[0.2, 0.68], [0.4, 0.26]]
    np.testing.assert_array_equal(np.round(combined_probabilities, decimals=2), combined_probabilities_expected)

    # One model, ndim = 3
    probabilities = np.array([[0.4, 0.6], [0.8, 0.2]])
    probabilities_exp = np.array(np.expand_dims(probabilities, axis=0))
    assert probabilities_exp.shape == (1, 2, 2)

    combined_probabilities = utils.combine_probabilities(probabilities_exp, 'maximum')
    assert combined_probabilities.shape == (2, 2)
    np.testing.assert_array_equal(combined_probabilities, probabilities)

    # One model, ndim=2
    probabilities = np.array([[0.4, 0.6], [0.8, 0.2]])
    assert probabilities.shape == (2, 2)
    combined_probabilities = utils.combine_probabilities(probabilities)
    assert combined_probabilities.shape == (2, 2)
    np.testing.assert_array_equal(combined_probabilities, probabilities)


def test_load_preprocess_image(test_image_path, model_spec_mobilenet):
    image = utils.load_preprocess_image(test_image_path, model_spec_mobilenet)
    assert image.shape == (1, 224, 224, 3)


def test_load_preprocess_images(test_folder_image_path, model_spec_mobilenet):
    images, images_paths = utils.load_preprocess_images(test_folder_image_path, model_spec_mobilenet)
    assert np.array(images).shape == (2, 224, 224, 3)
    assert len(images_paths) == 2


def test_create_concepts_default():
    concepts_by_default = utils.create_concepts_default(2)
    assert concepts_by_default == [{'label': 'Class_0', 'id': 'C_0'},
                                   {'label': 'Class_1', 'id': 'C_1'}]


def test_get_class_dictionaries_items():
    concepts_by_default = utils.create_concepts_default(2)
    output = utils.get_concept_items(concepts_by_default, 'label')
    assert output == ['Class_0', 'Class_1']

    output = utils.get_concept_items(concepts_by_default, 'id')
    assert output == ['C_0', 'C_1']
