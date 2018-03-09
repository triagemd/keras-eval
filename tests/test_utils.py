import keras_eval.utils as utils
import numpy as np
from keras.applications import mobilenet
import tensorflow as tf


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
    probs = np.random.rand(3, 100, 5)

    # Maximum
    probs_combined = utils.combine_probs(probs, 'maximum')
    assert len(probs_combined.shape) == 2

    # Arithmetic
    probs_combined = utils.combine_probs(probs, 'arithmetic')
    assert len(probs_combined.shape) == 2

    # Geometric
    probs_combined = utils.combine_probs(probs, 'geometric')
    assert len(probs_combined.shape) == 2

    # Harmonic
    probs_combined = utils.combine_probs(probs, 'harmonic')
    assert len(probs_combined.shape) == 2
