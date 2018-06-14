import json
import numpy as np
import os
import pytest

from keras_eval.eval import Evaluator


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


def check_evaluate_on_catdog_dataset(evaluator):
    probabilities, labels = evaluator.evaluate(os.path.abspath('tests/files/catdog/test'))

    # n_models x n_samples x n_classes
    assert len(probabilities.shape) == 3

    # n_samples x n_classes
    assert len(labels.shape) == 2

    # n_classes = 2
    assert evaluator.num_classes == 2

    # n_samples x n_classes
    assert len(evaluator.combined_probabilities.shape) == 2

    # class abbreviations
    assert evaluator.concept_labels == ['Class_0', 'Class_1']


def check_predict_on_cat_folder(evaluator):
    probabilities = evaluator.predict(os.path.abspath('tests/files/catdog/test/cat/'))

    # n_models x n_samples x n_classes
    assert len(probabilities.shape) == 3

    # 2 images in the folder
    assert len(evaluator.image_paths) == 2


def check_predict_single_image(evaluator):
    probabilities = evaluator.predict(os.path.abspath('tests/files/catdog/test/cat/cat-1.jpg'))

    # n_models x n_samples x n_classes
    assert len(probabilities.shape) == 3

    # 1 image predicted
    assert len(evaluator.image_paths) == 1


def test_get_image_paths_by_prediction(evaluator_mobilenet):
    probabilities, labels = evaluator_mobilenet.evaluate(os.path.abspath('tests/files/catdog/test'))
    image_paths_dictionary = evaluator_mobilenet.get_image_paths_by_prediction(probabilities, labels)

    assert image_paths_dictionary['Class_0_Class_0'] == [os.path.abspath('tests/files/catdog/test/cat/cat-1.jpg'),
                                                         os.path.abspath('tests/files/catdog/test/cat/cat-4.jpg')]
    assert image_paths_dictionary['Class_0_Class_1'] == []
    assert image_paths_dictionary['Class_1_Class_0'] == []
    assert image_paths_dictionary['Class_1_Class_1'] == [os.path.abspath('tests/files/catdog/test/dog/dog-2.jpg'),
                                                         os.path.abspath('tests/files/catdog/test/dog/dog-4.jpg')]


def test_evaluator_single_mobilenet_v1_on_catdog_dataset(evaluator_mobilenet):
    check_evaluate_on_catdog_dataset(evaluator_mobilenet)

    check_predict_on_cat_folder(evaluator_mobilenet)

    check_predict_single_image(evaluator_mobilenet)


def test_evaluator_ensemble_mobilenet_v1_on_catdog_dataset(evaluator_ensemble_mobilenet):
    check_evaluate_on_catdog_dataset(evaluator_ensemble_mobilenet)

    check_predict_on_cat_folder(evaluator_ensemble_mobilenet)

    check_predict_single_image(evaluator_ensemble_mobilenet)


def test_compute_confidence_prediction_distribution(evaluator_mobilenet):
    with pytest.raises(ValueError) as exception:
        evaluator_mobilenet.compute_confidence_prediction_distribution()
    expected = 'probabilities value is None, please run a evaluation first'
    actual = str(exception).split('ValueError: ')[1]
    assert actual == expected

    evaluator_mobilenet.evaluate(os.path.abspath('tests/files/catdog/test'))

    output = evaluator_mobilenet.compute_confidence_prediction_distribution()

    np.testing.assert_array_almost_equal(output, np.array([0.95398325, 0.0460167], dtype=np.float32))


def test_compute_uncertainty_distribution(evaluator_mobilenet):
    with pytest.raises(ValueError) as exception:
        evaluator_mobilenet.compute_uncertainty_distribution()
    expected = 'probabilities value is None, please run a evaluation first'
    actual = str(exception).split('ValueError: ')[1]
    assert actual == expected

    evaluator_mobilenet.evaluate(os.path.abspath('tests/files/catdog/test'))

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


def test_show_results(evaluator_mobilenet):
    evaluator_mobilenet.evaluate(os.path.abspath('tests/files/catdog/test'))

    average_df = evaluator_mobilenet.show_results('average')
    assert average_df['model'][0] == 'mobilenet_v1.h5'
    assert average_df['accuracy'][0] == average_df['precision'][0] == average_df['sensitivity'][0] \
        == average_df['f1_score'][0] == 1.0
    assert average_df['auroc'][0] == 0.833
    assert average_df['fdr'][0] == 0.0

    individual_df = evaluator_mobilenet.show_results('individual')
    assert individual_df['class'][0] == 'C_0'
    assert individual_df['class'][1] == 'C_1'
    assert individual_df['sensitivity'][0] == individual_df['sensitivity'][1] == 1.0
    assert individual_df['precision'][0] == individual_df['precision'][1] == 1.0
    assert individual_df['f1_score'][0] == individual_df['f1_score'][1] == 1.0
    assert individual_df['TP'][0] == individual_df['TP'][1] == 2
    assert individual_df['FP'][0] == individual_df['FP'][1] == individual_df['FN'][1] == individual_df['FN'][1] == 0
    assert individual_df['AUROC'][0] == individual_df['AUROC'][1] == 0.833
