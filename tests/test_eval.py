import os
import pytest
import numpy as np

from keras_eval import utils


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


def check_evaluate_on_catdog_dataset(evaluator, test_catdog_dataset_path):
    probabilities, labels = evaluator.evaluate(test_catdog_dataset_path)

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


def check_predict_single_image(evaluator, test_image_path):
    probabilities = evaluator.predict(test_image_path)

    # n_models x n_samples x n_classes
    assert len(probabilities.shape) == 3

    # 1 image predicted
    assert len(evaluator.image_paths) == 1


def test_check_compute_inference_probabilities_data_augmentation(evaluator_mobilenet_data_augmentation,
                                                                 test_catdog_dataset_path):
    probabilities, labels = evaluator_mobilenet_data_augmentation.evaluate(test_catdog_dataset_path)

    assert probabilities[0].shape == (4, 2)

    np.testing.assert_almost_equal(sum(sum(p[1] for p in probabilities)), 1.0)


def test_check_compute_inference_probabilities(evaluator_mobilenet_class_combine, test_animals_dataset_path):
    probabilities, labels = evaluator_mobilenet_class_combine.evaluate(test_animals_dataset_path)

    assert probabilities[0].shape == (15, 3)

    np.testing.assert_almost_equal(sum(sum(p[1] for p in probabilities)), 1.0)


def test_get_image_paths_by_prediction(evaluator_mobilenet, test_catdog_dataset_path, test_cat_folder, test_dog_folder):
    probabilities, labels = evaluator_mobilenet.evaluate(test_catdog_dataset_path)
    image_dictionary = evaluator_mobilenet.get_image_paths_by_prediction(probabilities, labels)

    assert image_dictionary['cat_cat']['image_paths'] == [os.path.join(test_cat_folder, 'cat-1.jpg'),
                                                          os.path.join(test_cat_folder, 'cat-4.jpg')]
    assert len(image_dictionary['cat_cat']['probs']) == 2
    assert image_dictionary['cat_dog']['image_paths'] == []
    assert len(image_dictionary['cat_dog']['probs']) == 0
    assert image_dictionary['dog_cat']['image_paths'] == [os.path.join(test_dog_folder, 'dog-2.jpg')]
    assert len(image_dictionary['dog_cat']['probs']) == 1
    assert image_dictionary['dog_dog']['image_paths'] == [os.path.join(test_dog_folder, 'dog-4.jpg')]
    assert len(image_dictionary['dog_dog']['probs']) == 1


def test_evaluator_single_mobilenet_v1_on_catdog_dataset(evaluator_mobilenet, test_catdog_dataset_path,
                                                         test_cat_folder, test_image_path):
    check_evaluate_on_catdog_dataset(evaluator_mobilenet, test_catdog_dataset_path)

    check_predict_on_cat_folder(evaluator_mobilenet, test_cat_folder)

    check_predict_single_image(evaluator_mobilenet, test_image_path)


def test_evaluator_ensemble_mobilenet_v1_on_catdog_dataset(evaluator_ensemble_mobilenet, test_catdog_dataset_path,
                                                           test_cat_folder, test_image_path):
    check_evaluate_on_catdog_dataset(evaluator_ensemble_mobilenet, test_catdog_dataset_path)

    check_predict_on_cat_folder(evaluator_ensemble_mobilenet, test_cat_folder)

    check_predict_single_image(evaluator_ensemble_mobilenet, test_image_path)


def test_compute_confidence_prediction_distribution(evaluator_mobilenet, test_catdog_dataset_path):
    with pytest.raises(ValueError) as exception:
        evaluator_mobilenet.compute_confidence_prediction_distribution()
    expected = 'probabilities value is None, please run an evaluation first'
    actual = str(exception).split('ValueError: ')[1]
    assert actual == expected

    evaluator_mobilenet.evaluate(test_catdog_dataset_path)

    output = evaluator_mobilenet.compute_confidence_prediction_distribution()

    np.testing.assert_array_almost_equal(output, np.array([0.82156974, 0.1784302], dtype=np.float32))


def test_compute_uncertainty_distribution(evaluator_mobilenet, test_catdog_dataset_path):
    with pytest.raises(ValueError) as exception:
        evaluator_mobilenet.compute_uncertainty_distribution()
    expected = 'probabilities value is None, please run an evaluation first'
    actual = str(exception).split('ValueError: ')[1]
    assert actual == expected

    evaluator_mobilenet.evaluate(test_catdog_dataset_path)

    output = evaluator_mobilenet.compute_uncertainty_distribution()

    np.testing.assert_array_almost_equal(output, np.array([0.8283282, 0.13131963, 0.36905038, 0.9456398], dtype=np.float32))


def test_plot_top_k_accuracy(evaluator_mobilenet):
    with pytest.raises(ValueError) as exception:
        evaluator_mobilenet.plot_top_k_accuracy()
    expected = 'results parameter is None, please run an evaluation first'
    actual = str(exception).split('ValueError: ')[1]
    assert actual == expected


def test_plot_top_k_sensitivity_by_concept(evaluator_mobilenet):
    with pytest.raises(ValueError) as exception:
        evaluator_mobilenet.plot_top_k_sensitivity_by_concept()
    expected = 'results parameter is None, please run an evaluation first'
    actual = str(exception).split('ValueError: ')[1]
    assert actual == expected


def test_show_results(evaluator_mobilenet, test_catdog_dataset_path):
    # Assert error without results
    with pytest.raises(ValueError) as exception:
        evaluator_mobilenet.show_results('average')
    expected = 'results parameter is None, please run an evaluation first'
    actual = str(exception).split('ValueError: ')[1]
    assert actual == expected

    evaluator_mobilenet.evaluate(test_catdog_dataset_path)

    average_df = evaluator_mobilenet.show_results(mode='average')
    assert average_df['id'][0] == 'catdog-mobilenet.hdf5'
    assert average_df['accuracy'][0] == 0.75
    assert average_df['sensitivity'][0] == 0.75
    assert average_df['weighted_precision'][0] == 0.833
    assert average_df['precision'][0] == 0.833
    assert average_df['f1_score'][0] == 0.733

    individual_df = evaluator_mobilenet.show_results(mode='individual')
    assert individual_df['class'][0] == 'cat'
    assert individual_df['class'][1] == 'dog'
    assert individual_df['sensitivity'][0] == 1.0
    assert individual_df['sensitivity'][1] == 0.5
    np.testing.assert_almost_equal(individual_df['precision'][0], 0.6669999)
    assert individual_df['precision'][1] == 1.0
    assert individual_df['f1_score'][0] == 0.8
    assert individual_df['f1_score'][1] == 0.667
    assert individual_df['TP'][0] == 2
    assert individual_df['TP'][1] == individual_df['FP'][0] == individual_df['FN'][1] == 1
    assert individual_df['FP'][1] == individual_df['FN'][0] == 0
    assert individual_df['AUROC'][0] == individual_df['AUROC'][1] == 1.0


def test_save_results(evaluator_mobilenet):
    # Assert error without results
    with pytest.raises(ValueError) as exception:
        evaluator_mobilenet.save_results('average', csv_path='')
    expected = 'results parameter is None, please run an evaluation first'
    actual = str(exception).split('ValueError: ')[1]
    assert actual == expected


def test_plot_probability_histogram(evaluator_mobilenet, test_catdog_dataset_path):
    with pytest.raises(ValueError) as exception:
        evaluator_mobilenet.plot_probability_histogram()
    expected = 'There are not computed probabilities. Please run an evaluation first.'
    actual = str(exception).split('ValueError: ')[1]
    assert actual == expected

    evaluator_mobilenet.evaluate(test_catdog_dataset_path)

    with pytest.raises(ValueError) as exception:
        evaluator_mobilenet.plot_probability_histogram(mode='x')
    expected = 'Incorrect mode. Supported modes are "errors" and "correct"'
    actual = str(exception).split('ValueError: ')[1]
    assert actual == expected


def test_plot_most_confident(evaluator_mobilenet, test_catdog_dataset_path):
    with pytest.raises(ValueError) as exception:
        evaluator_mobilenet.plot_most_confident()
    expected = 'There are not computed probabilities. Please run an evaluation first.'
    actual = str(exception).split('ValueError: ')[1]
    assert actual == expected

    evaluator_mobilenet.evaluate(test_catdog_dataset_path)

    with pytest.raises(ValueError) as exception:
        evaluator_mobilenet.plot_most_confident(mode='x')
    expected = 'Incorrect mode. Supported modes are "errors" and "correct"'
    actual = str(exception).split('ValueError: ')[1]
    assert actual == expected


def test_plot_confidence_interval(evaluator_mobilenet, test_catdog_dataset_path):
    with pytest.raises(ValueError) as exception:
        evaluator_mobilenet.plot_confidence_interval()
    expected = 'There are not computed probabilities. Please run an evaluation first.'
    actual = str(exception).split('ValueError: ')[1]
    assert actual == expected

    evaluator_mobilenet.evaluate(test_catdog_dataset_path)

    with pytest.raises(ValueError) as exception:
        evaluator_mobilenet.plot_confidence_interval(mode='x')
    expected = 'Incorrect mode. Modes available are "accuracy" or "error".'
    actual = str(exception).split('ValueError: ')[1]
    assert actual == expected


def test_plot_sensitivity_per_samples(evaluator_mobilenet):
    with pytest.raises(ValueError) as exception:
        evaluator_mobilenet.plot_sensitivity_per_samples()
    expected = 'results parameter is None, please run an evaluation first'
    actual = str(exception).split('ValueError: ')[1]
    assert actual == expected


def test_get_sensitivity_per_samples(evaluator_mobilenet, test_catdog_dataset_path):
    with pytest.raises(ValueError) as exception:
        evaluator_mobilenet.get_sensitivity_per_samples()
    expected = 'results parameter is None, please run an evaluation first'
    actual = str(exception).split('ValueError: ')[1]
    assert actual == expected

    evaluator_mobilenet.evaluate(test_catdog_dataset_path)
    results_classes = evaluator_mobilenet.get_sensitivity_per_samples()

    assert results_classes['sensitivity'][0] == 0.5
    assert results_classes['sensitivity'][1] == 1.0
    assert results_classes['class'][0] == 'dog'
    assert results_classes['class'][1] == 'cat'
    assert results_classes['% of samples'][0] == 50.0
    assert results_classes['% of samples'][1] == 50.0


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
