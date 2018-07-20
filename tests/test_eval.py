import json
import numpy as np
import os
import pytest

from keras_eval.eval import Evaluator
from keras_eval import utils


@pytest.fixture('session')
def test_dataset_path():
    return os.path.abspath(os.path.join('tests', 'files', 'catdog', 'test'))


@pytest.fixture('session')
def test_cat_folder():
    return os.path.abspath(os.path.join('tests', 'files', 'catdog', 'test', 'cat'))


@pytest.fixture('session')
def test_dog_folder():
    return os.path.abspath(os.path.join('tests', 'files', 'catdog', 'test', 'dog'))


@pytest.fixture('function')
def evaluator_mobilenet():
    specs = {'klass': 'keras.applications.mobilenet.MobileNet',
             'name': 'mobilenet_v1',
             'preprocess_args': None,
             'preprocess_func': 'between_plus_minus_1',
             'target_size': [224, 224, 3]
             }

    with open(os.path.abspath('tmp/fixtures/models/mobilenet_1/model_spec.json'), 'w') as outfile:
        json.dump(specs, outfile)

    return Evaluator(
        batch_size=1,
        model_path='tmp/fixtures/models/mobilenet_1/mobilenet_v1.h5'
    )


@pytest.fixture('function')
def evaluator_ensemble_mobilenet():
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

    return Evaluator(
        ensemble_models_dir='tmp/fixtures/models/',
        combination_mode='arithmetic',
        batch_size=1
    )


def test_set_concepts(evaluator_mobilenet):
    with pytest.raises(ValueError) as exception:
        evaluator_mobilenet.set_concepts([{'id': 'abcd', 'label': 'asd'}, {'a': 'b', 'b': 'c'}])
    expected = 'Incorrect format for concepts list. It must contain the fields `id` and `label`'
    actual = str(exception).split('ValueError: ')[1]
    assert actual == expected

    evaluator_mobilenet.set_concepts([{'id': '1', 'label': '1'}, {'id': '2', 'label': '2'}])


def test_set_combination_mode(evaluator_mobilenet):
    with pytest.raises(ValueError) as exception:
        evaluator_mobilenet.set_combination_mode('asdf')
    expected = 'Error: invalid option for `combination_mode` asdf'
    actual = str(exception).split('ValueError: ')[1]
    assert actual == expected

    evaluator_mobilenet.set_combination_mode('maximum')


def check_evaluate_on_catdog_dataset(evaluator, test_dataset_path):
    probabilities, labels = evaluator.evaluate(test_dataset_path)

    # n_models x n_samples x n_classes
    assert len(probabilities.shape) == 3

    # n_samples x n_classes
    assert len(labels.shape) == 2

    # n_classes = 2
    assert evaluator.num_classes == 2

    # n_samples x n_classes
    assert len(evaluator.combined_probabilities.shape) == 2

    # class abbreviations
    assert evaluator.concept_labels == ['cat', 'dog']


def check_predict_on_cat_folder(evaluator, test_cat_folder):
    probabilities = evaluator.predict(test_cat_folder)

    # n_models x n_samples x n_classes
    assert len(probabilities.shape) == 3

    # 2 images in the folder
    assert len(evaluator.image_paths) == 2


def check_predict_single_image(evaluator, test_cat_folder):
    probabilities = evaluator.predict(os.path.join(test_cat_folder, 'cat-1.jpg'))

    # n_models x n_samples x n_classes
    assert len(probabilities.shape) == 3

    # 1 image predicted
    assert len(evaluator.image_paths) == 1


def test_get_image_paths_by_prediction(evaluator_mobilenet, test_dataset_path, test_cat_folder, test_dog_folder):
    probabilities, labels = evaluator_mobilenet.evaluate(test_dataset_path)
    image_paths_dictionary = evaluator_mobilenet.get_image_paths_by_prediction(probabilities, labels)

    assert image_paths_dictionary['cat_cat'] == [os.path.join(test_cat_folder, 'cat-1.jpg'),
                                                 os.path.join(test_cat_folder, 'cat-4.jpg')]
    assert image_paths_dictionary['cat_dog'] == []
    assert image_paths_dictionary['dog_cat'] == []
    assert image_paths_dictionary['dog_dog'] == [os.path.join(test_dog_folder, 'dog-2.jpg'),
                                                 os.path.join(test_dog_folder, 'dog-4.jpg')]


def test_evaluator_single_mobilenet_v1_on_catdog_dataset(evaluator_mobilenet, test_dataset_path, test_cat_folder):
    check_evaluate_on_catdog_dataset(evaluator_mobilenet, test_dataset_path)

    check_predict_on_cat_folder(evaluator_mobilenet, test_cat_folder)

    check_predict_single_image(evaluator_mobilenet, test_cat_folder)


def test_evaluator_ensemble_mobilenet_v1_on_catdog_dataset(evaluator_ensemble_mobilenet, test_dataset_path, test_cat_folder):
    check_evaluate_on_catdog_dataset(evaluator_ensemble_mobilenet, test_dataset_path)

    check_predict_on_cat_folder(evaluator_ensemble_mobilenet, test_cat_folder)

    check_predict_single_image(evaluator_ensemble_mobilenet, test_cat_folder)


def test_compute_confidence_prediction_distribution(evaluator_mobilenet, test_dataset_path):
    with pytest.raises(ValueError) as exception:
        evaluator_mobilenet.compute_confidence_prediction_distribution()
    expected = 'probabilities value is None, please run a evaluation first'
    actual = str(exception).split('ValueError: ')[1]
    assert actual == expected

    evaluator_mobilenet.evaluate(test_dataset_path)

    output = evaluator_mobilenet.compute_confidence_prediction_distribution()

    np.testing.assert_array_almost_equal(output, np.array([0.95398325, 0.0460167], dtype=np.float32))


def test_compute_uncertainty_distribution(evaluator_mobilenet, test_dataset_path):
    with pytest.raises(ValueError) as exception:
        evaluator_mobilenet.compute_uncertainty_distribution()
    expected = 'probabilities value is None, please run a evaluation first'
    actual = str(exception).split('ValueError: ')[1]
    assert actual == expected

    evaluator_mobilenet.evaluate(test_dataset_path)

    output = evaluator_mobilenet.compute_uncertainty_distribution()

    np.testing.assert_array_almost_equal(output, np.array([0.3436, 0.002734, 0.001692, 0.52829], dtype=np.float32))


def test_plot_top_k_accuracy(evaluator_mobilenet):
    with pytest.raises(ValueError) as exception:
        evaluator_mobilenet.plot_top_k_accuracy()
    expected = 'results parameter is None, please run a evaluation first'
    actual = str(exception).split('ValueError: ')[1]
    assert actual == expected


def test_plot_top_k_sensitivity_by_concept(evaluator_mobilenet):
    with pytest.raises(ValueError) as exception:
        evaluator_mobilenet.plot_top_k_sensitivity_by_concept()
    expected = 'results parameter is None, please run a evaluation first'
    actual = str(exception).split('ValueError: ')[1]
    assert actual == expected


def test_show_results(evaluator_mobilenet, test_dataset_path):
    # Assert error without results
    with pytest.raises(ValueError) as exception:
        evaluator_mobilenet.show_results('average')
    expected = 'results parameter is None, please run a evaluation first'
    actual = str(exception).split('ValueError: ')[1]
    assert actual == expected

    evaluator_mobilenet.evaluate(test_dataset_path)

    average_df = evaluator_mobilenet.show_results(mode='average')
    assert average_df['model'][0] == 'mobilenet_v1.h5'
    assert average_df['accuracy'][0] == average_df['precision'][0] == average_df['f1_score'][0] == 1.0

    individual_df = evaluator_mobilenet.show_results(mode='individual')
    assert individual_df['class'][0] == 'cat'
    assert individual_df['class'][1] == 'dog'
    assert individual_df['sensitivity'][0] == individual_df['sensitivity'][1] == 1.0
    assert individual_df['precision'][0] == individual_df['precision'][1] == 1.0
    assert individual_df['f1_score'][0] == individual_df['f1_score'][1] == 1.0
    assert individual_df['TP'][0] == individual_df['TP'][1] == 2
    assert individual_df['FP'][0] == individual_df['FP'][1] == individual_df['FN'][1] == individual_df['FN'][1] == 0
    assert individual_df['AUROC'][0] == individual_df['AUROC'][1] == 1.0


def test_ensemble_models(evaluator_ensemble_mobilenet, test_cat_folder):
    ensemble = evaluator_ensemble_mobilenet.ensemble_models(input_shape=(224, 224, 3), combination_mode='average')
    model_spec = evaluator_ensemble_mobilenet.model_specs[0]
    image = utils.load_preprocess_image(os.path.join(test_cat_folder, 'cat-1.jpg'), model_spec)

    # forward pass
    preds = ensemble.predict(image)
    # 1 sample
    assert preds.shape[0] == 1
    # 2 predictions
    assert preds.shape[1] == 2
