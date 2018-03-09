import os
import json

from keras_eval.eval import Evaluator
from keras.applications import mobilenet
import tensorflow as tf


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
    assert evaluator.class_abbrevs == ['C_0', 'C_1']


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
