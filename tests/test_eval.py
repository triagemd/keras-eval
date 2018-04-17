import os
import json

from keras_eval.eval import Evaluator
from keras.applications import mobilenet
import tensorflow as tf
import numpy as np


def check_evaluate_on_catdog_datasets(eval_args={}):
    evaluator = Evaluator(
        data_dir=os.path.abspath('tests/files/catdog/test'),
        batch_size=1,
        **eval_args
    )

    probs, labels = evaluator.evaluate()

    # n_models x n_samples x n_classes
    assert len(probs.shape) == 3

    # n_samples x n_classes
    assert len(labels.shape) == 2

    # n_classes = 2
    assert evaluator.num_classes == 2

    # n_samples x n_classes
    assert len(evaluator.probs_combined.shape) == 2

    # class abbreviations
    assert evaluator.concept_labels == ['C_0', 'C_1']


def check_predict_on_cat_folder(eval_args={}):
    evaluator = Evaluator(
        data_dir=os.path.abspath('tests/files/catdog/test/cat/'),
        batch_size=1,
        **eval_args
    )

    probs = evaluator.predict()

    # n_models x n_samples x n_classes
    assert len(probs.shape) == 3

    # 2 images in the folder
    assert len(evaluator.image_paths) == 2


def check_predict_single_image(eval_args={}):
    evaluator = Evaluator(
        data_dir=os.path.abspath('tests/files/catdog/test/cat/cat-1.jpg'),
        batch_size=1,
        **eval_args
    )

    probs = evaluator.predict()

    # n_models x n_samples x n_classes
    assert len(probs.shape) == 3

    # 1 image predicted
    assert len(evaluator.image_paths) == 1


def test_get_image_paths_by_prediction():
    specs = {'klass': 'keras.applications.mobilenet.MobileNet',
             'name': 'mobilenet_v1',
             'preprocess_args': None,
             'preprocess_func': 'between_plus_minus_1',
             'target_size': [224, 224, 3]
             }

    with open(os.path.abspath('tmp/fixtures/models/mobilenet_1/model_spec.json'), 'w') as outfile:
        json.dump(specs, outfile)

    evaluator = Evaluator(
        data_dir=os.path.abspath('tests/files/catdog/test'),
        batch_size=1,
        model_path='tmp/fixtures/models/mobilenet_1/mobilenet_v1.h5'
    )

    probs, labels = evaluator.evaluate()
    image_paths_dictionary = evaluator.get_image_paths_by_prediction(probs, labels)

    assert image_paths_dictionary['C_0_C_0'] == [os.path.abspath('tests/files/catdog/test/cat/cat-1.jpg'),
                                                 os.path.abspath('tests/files/catdog/test/cat/cat-4.jpg')]
    assert image_paths_dictionary['C_0_C_1'] == []
    assert image_paths_dictionary['C_1_C_0'] == []
    assert image_paths_dictionary['C_1_C_1'] == [os.path.abspath('tests/files/catdog/test/dog/dog-2.jpg'),
                                                 os.path.abspath('tests/files/catdog/test/dog/dog-4.jpg')]


def test_evaluator_mobilenet_v1_on_catdog_dataset():
    specs = {'klass': 'keras.applications.mobilenet.MobileNet',
             'name': 'mobilenet_v1',
             'preprocess_args': None,
             'preprocess_func': 'between_plus_minus_1',
             'target_size': [224, 224, 3]
             }

    with open(os.path.abspath('tmp/fixtures/models/mobilenet_1/model_spec.json'), 'w') as outfile:
        json.dump(specs, outfile)

    custom_objects = {'relu6': mobilenet.relu6, 'DepthwiseConv2D': mobilenet.DepthwiseConv2D, "tf": tf}

    eval_options = {'custom_objects': custom_objects, 'model_path': 'tmp/fixtures/models/mobilenet_1/mobilenet_v1.h5'}

    check_evaluate_on_catdog_datasets(eval_options)

    check_predict_on_cat_folder(eval_options)

    check_predict_single_image(eval_options)


def test_evaluator_ensemble_mobilenet_v1_on_catdog_dataset():
    specs = {'klass': 'keras.applications.mobilenet.MobileNet',
             'name': 'mobilenet_v1',
             'preprocess_args': None,
             'preprocess_func': 'between_plus_minus_1',
             'target_size': [224, 224, 3]
             }

    with open(os.path.abspath('tmp/fixtures/models/mobilenet_1/model_spec.json'), 'w') as outfile:
        json.dump(specs, outfile)

    with open(os.path.abspath('tmp/fixtures/models/mobilenet_2/model_spec.json'), 'w') as outfile:
        json.dump(specs, outfile)

    eval_options = {'ensemble_models_dir': 'tmp/fixtures/models/'}

    check_evaluate_on_catdog_datasets(eval_options)

    check_predict_on_cat_folder(eval_options)

    check_predict_single_image(eval_options)


def test_compute_confidence_prediction_distribution():
    specs = {'klass': 'keras.applications.mobilenet.MobileNet',
             'name': 'mobilenet_v1',
             'preprocess_args': None,
             'preprocess_func': 'between_plus_minus_1',
             'target_size': [224, 224, 3]
             }

    with open(os.path.abspath('tmp/fixtures/models/mobilenet_1/model_spec.json'), 'w') as outfile:
        json.dump(specs, outfile)

    evaluator = Evaluator(
        data_dir=os.path.abspath('tests/files/catdog/test'),
        batch_size=1,
        model_path='tmp/fixtures/models/mobilenet_1/mobilenet_v1.h5'
    )

    probs, labels = evaluator.evaluate()

    output = evaluator.compute_confidence_prediction_distribution()

    np.testing.assert_array_almost_equal(output, np.array([0.95398325, 0.0460167], dtype=np.float32))


def test_compute_uncertainty_distribution():
    specs = {'klass': 'keras.applications.mobilenet.MobileNet',
             'name': 'mobilenet_v1',
             'preprocess_args': None,
             'preprocess_func': 'between_plus_minus_1',
             'target_size': [224, 224, 3]
             }

    with open(os.path.abspath('tmp/fixtures/models/mobilenet_1/model_spec.json'), 'w') as outfile:
        json.dump(specs, outfile)

    evaluator = Evaluator(
        data_dir=os.path.abspath('tests/files/catdog/test'),
        batch_size=1,
        model_path='tmp/fixtures/models/mobilenet_1/mobilenet_v1.h5'
    )

    probs, labels = evaluator.evaluate()

    output = evaluator.compute_uncertainty_distribution()

    np.testing.assert_array_almost_equal(output, np.array([0.3436, 0.002734, 0.001692, 0.52829], dtype=np.float32))
