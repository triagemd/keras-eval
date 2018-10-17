import os
import pytest

from keras_eval.eval import Evaluator
from keras_model_specs import ModelSpec
from collections import OrderedDict


@pytest.fixture('session')
def test_catdog_dataset_path():
    return os.path.abspath(os.path.join('tests', 'files', 'catdog', 'test'))


@pytest.fixture('session')
def test_animals_dataset_path():
    return os.path.abspath(os.path.join('tests', 'files', 'animals', 'test'))


@pytest.fixture('session')
def test_cat_folder():
    return os.path.abspath(os.path.join('tests', 'files', 'catdog', 'test', 'cat'))


@pytest.fixture('session')
def test_dog_folder():
    return os.path.abspath(os.path.join('tests', 'files', 'catdog', 'test', 'dog'))


@pytest.fixture('session')
def training_dict_file():
    return os.path.abspath(os.path.join('tests', 'files', 'animals', 'dictionary.json'))


@pytest.fixture('session')
def test_image_path():
    return os.path.abspath(os.path.join('tests', 'files', 'catdog', 'test', 'cat', 'cat-1.jpg'))


@pytest.fixture('session')
def test_ensemble_models_path():
    return os.path.abspath(os.path.join('tmp', 'fixtures', 'models', 'ensemble'))


@pytest.fixture('session')
def test_catdog_mobilenet_model():
    return os.path.abspath(
        os.path.join('tmp', 'fixtures', 'models', 'ensemble', 'mobilenet_1', 'catdog-mobilenet.hdf5'))


@pytest.fixture('session')
def test_mobilenet_1_model_spec():
    return os.path.abspath(os.path.join('tmp', 'fixtures', 'models', 'ensemble', 'mobilenet_1', 'model_spec.json'))


@pytest.fixture('session')
def test_animals_model_path():
    return os.path.abspath(os.path.join('tmp', 'fixtures', 'models', 'single', 'animals-mobilenet.hdf5'))


@pytest.fixture('session')
def test_animals_dictionary_path():
    return os.path.abspath(os.path.join('tests', 'files', 'animals', 'dictionary.json'))


@pytest.fixture('session')
def model_spec_mobilenet():
    dataset_mean = [142.69182214, 119.05833338, 106.89884415]
    return ModelSpec.get('mobilenet_v1', preprocess_func='mean_subtraction', preprocess_args=dataset_mean)


@pytest.fixture('function')
def evaluator_mobilenet(test_catdog_mobilenet_model):
    return Evaluator(
        batch_size=1,
        model_path=test_catdog_mobilenet_model
    )


@pytest.fixture('function')
def evaluator_ensemble_mobilenet(test_ensemble_models_path):
    return Evaluator(
        ensemble_models_dir=test_ensemble_models_path,
        combination_mode='arithmetic',
        batch_size=1
    )


@pytest.fixture('function')
def evaluator_mobilenet_class_combine(test_animals_model_path, test_animals_dictionary_path):
    return Evaluator(
        batch_size=1,
        model_path=test_animals_model_path,
        concept_dictionary_path=test_animals_dictionary_path
    )
