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


@pytest.fixture('function')
def evaluator_results():
    results_1 = {'average': OrderedDict([('accuracy', [0.5, 1.0]), ('precision', 0.65), ('f1_score', 0.55),
                                    ('number_of_samples', 2000), ('number_of_classes', 2), ('confusion_matrix',
                                                                                            array([[907,  93],
                                                                                                   [741, 259]]))]),
            'individual': [{'concept': 'cats', 'metrics': OrderedDict([('sensitivity', 0.90), ('precision', 0.55),
                                                                       ('f1_score', 0.68), ('specificity', 0.25),
                                                                       ('FDR', 0.45), ('AUROC', 1.0), ('TP', 907),
                                                                       ('FP', 741), ('FN', 93), ('% of samples', 50.0)])
                            },
                           {'concept': 'dogs',
                            'metrics': OrderedDict([('sensitivity', 0.25), ('precision', 0.75),
                                                    ('f1_score', 0.4), ('specificity', 0.90), ('FDR', 0.25),
                                                    ('AUROC', 1.0), ('TP', 259), ('FP', 93), ('FN', 741),
                                                    ('% of samples', 50.0)])}]}
    results_2 = {'average': OrderedDict([('accuracy', [0.3, 1.0]), ('precision', 0.8), ('f1_score', 0.55),
                                         ('number_of_samples', 2000), ('number_of_classes', 2), ('confusion_matrix',
                                                                                                 array([[907, 93],
                                                                                                        [741, 259]]))]),
                 'individual': [{'concept': 'cats', 'metrics': OrderedDict([('sensitivity', 0.70), ('precision', 0.25),
                                                                            ('f1_score', 0.68), ('specificity', 0.25),
                                                                            ('FDR', 0.45), ('AUROC', 1.0), ('TP', 907),
                                                                            ('FP', 741), ('FN', 93),
                                                                            ('% of samples', 50.0)])
                                 },
                                {'concept': 'dogs',
                                 'metrics': OrderedDict([('sensitivity', 0.85), ('precision', 0.15),
                                                         ('f1_score', 0.4), ('specificity', 0.90), ('FDR', 0.25),
                                                         ('AUROC', 1.0), ('TP', 259), ('FP', 93), ('FN', 741),
                                                         ('% of samples', 50.0)])}]}
    return results_1, results_2
