import os
import pytest
import numpy as np
import keras_eval.utils as utils


def test_safe_divide():
    assert np.isnan(utils.safe_divide(10.0, 0.0))
    assert utils.safe_divide(10.0, 5.0) == 2.0


def test_round_list():
    input_list = [0.6666666666, 0.3333333333]
    assert utils.round_list(input_list, decimals=2) == [0.67, 0.33]
    assert utils.round_list(input_list, decimals=4) == [0.6667, 0.3333]
    assert utils.round_list(input_list, decimals=6) == [0.666667, 0.333333]


def test_read_dictionary(training_dict_file):
    dictionary = utils.read_dictionary(training_dict_file)
    expected = 5
    actual = len(dictionary)
    assert actual == expected


def test_load_model(test_catdog_mobilenet_model, test_mobilenet_2_model_spec):

    # Default model_spec
    model = utils.load_model(test_catdog_mobilenet_model)
    assert model

    # Custom model_spec
    model = utils.load_model(test_catdog_mobilenet_model, specs_path=test_mobilenet_2_model_spec)
    assert model


def test_load_model_ensemble(test_ensemble_models_path):
    models, specs = utils.load_multi_model(test_ensemble_models_path)
    assert models
    assert specs


def test_combine_probabilities():
    # Ensemble 3 models
    probabilities = [[[0.4, 0.6], [0.8, 0.2]], [[0.1, 0.9], [0.2, 0.6]], [[0.4, 0.6], [0.8, 0.2]]]

    # Maximum
    combined_probabilities = utils.combine_probabilities(probabilities, 'maximum')
    assert len(combined_probabilities.shape) == 2
    combined_probabilities_expected = [[0.4, 0.9], [0.8, 0.6]]
    np.testing.assert_array_equal(np.round(combined_probabilities, decimals=2), combined_probabilities_expected)

    # Arithmetic
    combined_probabilities = utils.combine_probabilities(probabilities, 'arithmetic')
    assert len(combined_probabilities.shape) == 2
    combined_probabilities_expected = [[0.3, 0.7], [0.6, 0.33]]
    np.testing.assert_array_equal(np.round(combined_probabilities, decimals=2), combined_probabilities_expected)

    # Geometric
    combined_probabilities = utils.combine_probabilities(probabilities, 'geometric')
    assert len(combined_probabilities.shape) == 2
    combined_probabilities_expected = [[0.25, 0.69], [0.5, 0.29]]
    np.testing.assert_array_equal(np.round(combined_probabilities, decimals=2), combined_probabilities_expected)

    # Harmonic
    combined_probabilities = utils.combine_probabilities(probabilities, 'harmonic')
    assert len(combined_probabilities.shape) == 2
    combined_probabilities_expected = [[0.2, 0.68], [0.4, 0.26]]
    np.testing.assert_array_equal(np.round(combined_probabilities, decimals=2), combined_probabilities_expected)

    # One model, ndim = 3
    probabilities = np.array([[0.4, 0.6], [0.8, 0.2]])
    probabilities_exp = np.array(np.expand_dims(probabilities, axis=0))
    assert probabilities_exp.shape == (1, 2, 2)

    combined_probabilities = utils.combine_probabilities(probabilities_exp, 'maximum')
    assert combined_probabilities.shape == (2, 2)
    np.testing.assert_array_equal(combined_probabilities, probabilities)

    # One model, ndim=2
    probabilities = np.array([[0.4, 0.6], [0.8, 0.2]])
    assert probabilities.shape == (2, 2)
    combined_probabilities = utils.combine_probabilities(probabilities)
    assert combined_probabilities.shape == (2, 2)
    np.testing.assert_array_equal(combined_probabilities, probabilities)


def test_load_preprocess_image(test_image_path, model_spec_mobilenet):
    image = utils.load_preprocess_image(test_image_path, model_spec_mobilenet)
    assert image.shape == (1, 224, 224, 3)


def test_load_preprocess_images(test_cat_folder, model_spec_mobilenet):
    images, images_paths = utils.load_preprocess_images(test_cat_folder, model_spec_mobilenet)
    assert np.array(images).shape == (2, 224, 224, 3)
    assert len(images_paths) == 2


def test_default_concepts(test_catdog_dataset_path):
    concepts_by_default = utils.get_default_concepts(test_catdog_dataset_path)
    assert concepts_by_default == [{'label': 'cat', 'id': 'cat'},
                                   {'label': 'dog', 'id': 'dog'}]


def test_create_training_json(test_catdog_dataset_path):
    dict_path = './tests/files/dict.json'
    utils.create_training_json(test_catdog_dataset_path, dict_path)
    actual = os.path.isfile(dict_path)
    expected = True
    assert actual == expected


def test_compare_concept_dictionaries():
    concept_lst = ['dog', 'elephant']
    concept_dict = [{'group': 'dog'}, {'group': 'cat'}, {'group': 'elephant'}]
    with pytest.raises(ValueError) as exception:
        utils.compare_group_test_concepts(concept_lst, concept_dict)
    expected = "('The following concepts are not present in the either the concept dictionary or among the " \
               "test classes:', ['cat'])"
    actual = str(exception).split('ValueError: ')[1]
    assert actual == expected


def test_check_concept_unique():
    concept_dict = [{'class_name': 'cat'}, {'class_name': 'dog'}, {'class_name': 'cat'}]
    with pytest.raises(ValueError) as exception:
        utils.check_concept_unique(concept_dict)
    expected = "('Concept has been repeated:', 'cat')"
    actual = str(exception).split('ValueError: ')[1]
    assert actual == expected


def test_get_class_dictionaries_items(test_catdog_dataset_path):
    concepts_by_default = utils.get_default_concepts(test_catdog_dataset_path)
    label_output = utils.get_concept_items(concepts_by_default, 'label')
    id_output = utils.get_concept_items(concepts_by_default, 'id')
    assert label_output == id_output == ['cat', 'dog']


def test_show_results():
    results = {'individual':
               [{'concept':
                 'Class_0', 'metrics': {'TP': 2, 'precision': 1.0, 'AUROC': 0.8333333, 'sensitivity': 1.0,
                                        'FN': 0, 'FDR': 0.0, 'f1_score': 1.0, 'FP': 0}},
                {'concept': 'Class_1', 'metrics': {'TP': 2, 'precision': 1.0, 'AUROC': 0.8333333,
                                                   'sensitivity': 1.0, 'FN': 0, 'FDR': 0.0,
                                                   'f1_score': 1.0, 'FP': 0}}],
               'average': {'precision': [1.0], 'confusion_matrix': np.array([[2, 0], [0, 2]]), 'sensitivity': [1.0],
                           'auroc': [0.8333333], 'f1_score': [1.0], 'accuracy': [1.0],
                           'specificity': [1.0], 'fdr': [0.0]}}

    concepts = [{'id': 'C_0', 'label': 'Class_0'}, {'id': 'C_1', 'label': 'Class_1'}]
    # Assert error when incorrect mode
    with pytest.raises(ValueError) as exception:
        utils.show_results(results, concepts, mode='asdf')
    expected = 'results mode must be either "average" or "individual"'
    actual = str(exception).split('ValueError: ')[1]
    assert actual == expected

    average_df = utils.show_results(results, concepts)
    assert average_df['model'][0] == 'default_model'
    assert average_df['accuracy'][0] == average_df['precision'][0] == average_df['f1_score'][0] == 1.0

    individual_df = utils.show_results(results, concepts, mode='individual')
    assert individual_df['class'][0] == 'C_0'
    assert individual_df['class'][1] == 'C_1'
    assert individual_df['sensitivity'][0] == individual_df['sensitivity'][1] == 1.0
    assert individual_df['precision'][0] == individual_df['precision'][1] == 1.0
    assert individual_df['f1_score'][0] == individual_df['f1_score'][1] == 1.0
    assert individual_df['TP'][0] == individual_df['TP'][1] == 2
    assert individual_df['FP'][0] == individual_df['FP'][1] == individual_df['FN'][1] == individual_df['FN'][1] == 0
    assert individual_df['AUROC'][0] == individual_df['AUROC'][1] == 0.833


def test_ensemble_models(test_image_path, model_spec_mobilenet, test_ensemble_models_path):
    models, model_specs = utils.load_multi_model(test_ensemble_models_path)

    with pytest.raises(ValueError) as exception:
        utils.ensemble_models(models, input_shape=(224, 224, 3), combination_mode='asdf')
    expected = 'Incorrect combination mode selected, we only allow for `average` or `maximum`'
    actual = str(exception).split('ValueError: ')[1]
    assert actual == expected

    with pytest.raises(ValueError) as exception:
        utils.ensemble_models(models, input_shape=(224, 3), combination_mode='asdf')
    expected = 'Incorrect input shape, it should have 3 dimensions (H, W, C)'
    actual = str(exception).split('ValueError: ')[1]
    assert actual == expected

    ensemble = utils.ensemble_models(models, input_shape=(224, 224, 3), combination_mode='average')
    image = utils.load_preprocess_image(test_image_path, model_spec_mobilenet)

    # forward pass
    preds = ensemble.predict(image)
    # 1 sample
    assert preds.shape[0] == 1
    # 2 predictions
    assert preds.shape[1] == 2
