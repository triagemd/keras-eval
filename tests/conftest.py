import os
import pytest
import numpy as np

from keras_eval.eval import Evaluator
from keras_model_specs import ModelSpec


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
def test_image_path():
    return os.path.abspath(os.path.join('tests', 'files', 'catdog', 'test', 'cat', 'cat-1.jpg'))


@pytest.fixture('session')
def test_catdog_ensemble_path():
    return os.path.abspath(os.path.join('tmp', 'fixtures', 'models', 'catdog'))


@pytest.fixture('session')
def test_catdog_mobilenet_model():
    return os.path.abspath(
        os.path.join('tmp', 'fixtures', 'models', 'catdog', 'mobilenet_1', 'catdog-mobilenet.hdf5'))


@pytest.fixture('session')
def test_catdog_mobilenet_model_spec():
    return os.path.abspath(os.path.join('tmp', 'fixtures', 'models', 'catdog', 'mobilenet_1', 'model_spec.json'))


@pytest.fixture('session')
def test_animals_ensemble_path():
    return os.path.abspath(os.path.join('tmp', 'fixtures', 'models', 'animals'))


@pytest.fixture('session')
def test_animals_mobilenet_path():
    return os.path.abspath(os.path.join('tmp', 'fixtures', 'models', 'animals', 'mobilenet_1', 'animals-mobilenet.hdf5'))


@pytest.fixture('session')
def test_animals_mobilenet_dictionary_path():
    return os.path.abspath(os.path.join('tests', 'files', 'animals', 'dictionary.json'))


@pytest.fixture('session')
def test_animals_dictionary_path():
    return os.path.abspath(os.path.join('tests', 'files', 'animals', 'dictionary.json'))


@pytest.fixture('session')
def test_average_results_csv_paths():
    return [os.path.abspath(os.path.join('tests', 'files', 'catdog', 'results_csv', 'eval_avg_1.csv')),
            os.path.abspath(os.path.join('tests', 'files', 'catdog', 'results_csv', 'eval_avg_2.csv')),
            os.path.abspath(os.path.join('tests', 'files', 'catdog', 'results_csv', 'eval_avg_3.csv'))]


@pytest.fixture('session')
def test_individual_results_csv_paths():
    return [os.path.abspath(os.path.join('tests', 'files', 'catdog', 'results_csv', 'eval_class_1.csv')),
            os.path.abspath(os.path.join('tests', 'files', 'catdog', 'results_csv', 'eval_class_2.csv')),
            os.path.abspath(os.path.join('tests', 'files', 'catdog', 'results_csv', 'eval_class_3.csv'))]


@pytest.fixture('session')
def test_results_csv_paths():
    return os.path.abspath(os.path.join('tests', 'files', 'eval'))


@pytest.fixture('session')
def model_spec_mobilenet():
    dataset_mean = [142.69182214, 119.05833338, 106.89884415]
    return ModelSpec.get('mobilenet_v1', preprocess_func='mean_subtraction', preprocess_args=dataset_mean)


@pytest.fixture('function')
def evaluator_catdog_mobilenet(test_catdog_mobilenet_model):
    return Evaluator(
        batch_size=1,
        model_path=test_catdog_mobilenet_model
    )


@pytest.fixture('function')
def evaluator_catdog_mobilenet_data_augmentation(test_catdog_mobilenet_model):
    return Evaluator(
        batch_size=1,
        model_path=test_catdog_mobilenet_model,
        data_augmentation={'scale_sizes': [256], 'transforms': ['horizontal_flip'], 'crop_original': 'center_crop'}
    )


@pytest.fixture('function')
def evaluator_catdog_ensemble(test_catdog_ensemble_path):
    return Evaluator(
        ensemble_models_dir=test_catdog_ensemble_path,
        combination_mode='arithmetic',
        batch_size=1
    )


@pytest.fixture('function')
def evaluator_animals_mobilenet_class_inference(test_animals_mobilenet_path, test_animals_dictionary_path):
    return Evaluator(
        model_path=test_animals_mobilenet_path,
        concept_dictionary_path=test_animals_dictionary_path,
        batch_size=1
    )


@pytest.fixture('function')
def evaluator_animals_ensemble_class_inference(test_animals_ensemble_path, test_animals_dictionary_path):
    return Evaluator(
        ensemble_models_dir=test_animals_ensemble_path,
        concept_dictionary_path=test_animals_dictionary_path,
        combination_mode='arithmetic',
        batch_size=1
    )


@pytest.fixture('session')
def metrics_top_k_binary_class():
    concepts = ['class0', 'class1']
    y_true = np.asarray([0, 1, 0, 1])  # 4 samples, 2 classes.
    y_probs = np.asarray([[1, 0], [0.2, 0.8], [0.8, 0.2], [0.35, 0.65]])
    return concepts, y_true, y_probs


@pytest.fixture('session')
def metrics_top_k_multi_class():
    concepts = ['class0', 'class1', 'class3']
    y_true = np.asarray([0, 1, 2, 2])  # 4 samples, 3 classes.
    y_probs = np.asarray([[1, 0, 0], [0.2, 0.2, 0.6], [0.8, 0.2, 0], [0.35, 0.25, 0.4]])
    return concepts, y_true, y_probs
