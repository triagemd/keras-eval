import keras_eval.utils as utils
import numpy as np
from keras.applications import mobilenet
import tensorflow as tf
import pytest
import os
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


def test_combine_probs():
    # Ensemble 3 models
    probs = [[[0.4, 0.6], [0.8, 0.2]], [[0.1, 0.9], [0.2, 0.6]], [[0.4, 0.6], [0.8, 0.2]]]

    # Maximum
    probs_combined = utils.combine_probabilities(probs, 'maximum')
    assert len(probs_combined.shape) == 2
    probs_combined_expected = [[0.4, 0.9], [0.8, 0.6]]
    np.testing.assert_array_equal(np.round(probs_combined, decimals=2), np.array(probs_combined_expected))

    # Arithmetic
    probs_combined = utils.combine_probabilities(probs, 'arithmetic')
    assert len(probs_combined.shape) == 2
    probs_combined_expected = [[0.3, 0.7], [0.6,  0.33]]
    np.testing.assert_array_equal(np.round(probs_combined, decimals=2), np.array(probs_combined_expected))

    # Geometric
    probs_combined = utils.combine_probabilities(probs, 'geometric')
    assert len(probs_combined.shape) == 2
    probs_combined_expected = [[0.25, 0.69], [0.5, 0.29]]
    np.testing.assert_array_equal(np.round(probs_combined, decimals=2), np.array(probs_combined_expected))

    # Harmonic
    probs_combined = utils.combine_probabilities(probs, 'harmonic')
    assert len(probs_combined.shape) == 2
    probs_combined_expected = [[0.2, 0.68], [0.4, 0.26]]
    np.testing.assert_array_equal(np.round(probs_combined, decimals=2), np.array(probs_combined_expected))

    # One model, ndim = 3
    probs = np.array([[0.4, 0.6], [0.8, 0.2]])
    probs_exp = np.array(np.expand_dims(probs, axis=0))
    assert probs_exp.shape == (1, 2, 2)

    probs_combined = utils.combine_probabilities(probs_exp, 'maximum')
    assert probs_combined.shape == (2, 2)
    np.testing.assert_array_equal(probs_combined, np.array(probs))

    # One model, ndim=2
    probs = np.array([[0.4, 0.6], [0.8, 0.2]])
    assert probs.shape == (2, 2)
    probs_combined = utils.combine_probabilities(probs)
    assert probs_combined.shape == (2, 2)
    np.testing.assert_array_equal(probs_combined, np.array(probs))


def test_load_preprocess_image(test_image_path, model_spec_mobilenet):
    image = utils.load_preprocess_image(test_image_path, model_spec_mobilenet)
    assert image.shape == (1, 224, 224, 3)


def test_load_preprocess_images(test_folder_image_path, model_spec_mobilenet):
    images, images_paths = utils.load_preprocess_images(test_folder_image_path, model_spec_mobilenet)
    assert np.array(images).shape == (2, 224, 224, 3)
    assert len(images_paths) == 2


def test_create_class_dictionary_default():
    class_dictionary_default = utils.create_class_dictionary_default(2)
    assert class_dictionary_default == [{'abbrev': 'C_0', 'class_name': 'Class_ 0'},
                                        {'abbrev': 'C_1', 'class_name': 'Class_ 1'}]


def test_get_class_dictionaries_items():
    class_dictionary_default = utils.create_class_dictionary_default(2)
    output = utils.get_class_dictionaries_items(class_dictionary_default, 'abbrev')
    assert output == ['C_0', 'C_1']

    output = utils.get_class_dictionaries_items(class_dictionary_default, 'class_name')
    assert output == ['Class_ 0', 'Class_ 1']
