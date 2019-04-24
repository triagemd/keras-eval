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


def test_read_dictionary(test_animals_dictionary_path):
    dictionary = utils.read_dictionary(test_animals_dictionary_path)
    expected = 5
    actual = len(dictionary)
    assert actual == expected


def test_load_model(test_catdog_mobilenet_model, test_catdog_mobilenet_model_spec):

    # Default model_spec
    model = utils.load_model(test_catdog_mobilenet_model)
    assert model

    # Custom model_spec
    model = utils.load_model(test_catdog_mobilenet_model, specs_path=test_catdog_mobilenet_model_spec)
    assert model


def test_load_model_ensemble(test_catdog_ensemble_path):
    models, specs = utils.load_multi_model(test_catdog_ensemble_path)
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


def test_get_dictionary_concepts(test_animals_dictionary_path):
    dictionary_concepts = utils.get_dictionary_concepts(test_animals_dictionary_path)
    assert dictionary_concepts == [{'label': '00000_cat', 'id': 0},
                                   {'label': '00001_dog', 'id': 1},
                                   {'label': '00002_goose', 'id': 2},
                                   {'label': '00003_turtle', 'id': 3},
                                   {'label': '00004_elephant', 'id': 4}]


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
    expected = "('The following concepts are not present in either the concept dictionary or among the " \
               "test classes:', ['cat'])"
    actual = str(exception).split('ValueError: ')[1]
    assert actual == expected


def test_check_input_samples(metrics_top_k_multi_class):
    _, y_true, y_probs = metrics_top_k_multi_class
    assert utils.check_input_samples(y_probs, y_true)

    y_true = np.asarray([0, 1, 2, 2, 1])  # 5 samples, 3 classes.
    y_probs = np.asarray([[1, 0, 0], [0.2, 0.2, 0.6], [0.8, 0.2, 0], [0.35, 0.25, 0.4]])  # 4 samples, 3 classes.
    with pytest.raises(ValueError) as exception:
        utils.check_input_samples(y_probs, y_true)
    expected = 'The number predicted samples (4) is different from the ground truth samples (5)'
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


def test_results_to_dataframe():
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

    # Assert error when incorrect mode
    with pytest.raises(ValueError) as exception:
        utils.results_to_dataframe(results, mode='asdf')
    expected = 'Results mode must be either "average" or "individual"'
    actual = str(exception).split('ValueError: ')[1]
    assert actual == expected

    average_df = utils.results_to_dataframe(results)
    assert average_df['id'][0] == 'default_model'
    assert average_df['accuracy'][0] == average_df['precision'][0] == average_df['f1_score'][0] == 1.0

    individual_df = utils.results_to_dataframe(results, mode='individual')
    assert individual_df['class'][0] == 'Class_0'
    assert individual_df['class'][1] == 'Class_1'
    assert individual_df['sensitivity'][0] == individual_df['sensitivity'][1] == 1.0
    assert individual_df['precision'][0] == individual_df['precision'][1] == 1.0
    assert individual_df['f1_score'][0] == individual_df['f1_score'][1] == 1.0
    assert individual_df['TP'][0] == individual_df['TP'][1] == 2
    assert individual_df['FP'][0] == individual_df['FP'][1] == individual_df['FN'][1] == individual_df['FN'][1] == 0
    assert individual_df['AUROC'][0] == individual_df['AUROC'][1] == 0.833


def test_ensemble_models(test_image_path, model_spec_mobilenet, test_catdog_ensemble_path):
    models, model_specs = utils.load_multi_model(test_catdog_ensemble_path)

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


def test_load_csv_to_dataframe(test_average_results_csv_paths):
    # No error
    assert len(utils.load_csv_to_dataframe(test_average_results_csv_paths)) == 3

    # Format error
    with pytest.raises(ValueError) as exception:
        utils.load_csv_to_dataframe(1)
    expected = 'Incorrect format for `csv_paths`, a list of strings or a single string are expected'
    actual = str(exception).split('ValueError: ')[1]
    assert actual == expected


def test_results_differential(test_average_results_csv_paths, test_individual_results_csv_paths):
    with pytest.raises(ValueError) as exception:
        utils.results_differential(['asd'])
    expected = 'The number of dataframes should be higher than 1'
    actual = str(exception).split('ValueError: ')[1]
    assert actual == expected

    dataframes = utils.load_csv_to_dataframe(test_average_results_csv_paths)
    df = utils.results_differential(dataframes, mode='average')
    assert len(df) == 3
    assert df['accuracy'][1] == '0.2 (-0.6)'
    assert df['accuracy'][2] == '0.9 (+0.1)'
    assert df['weighted_precision'][1] == '0.2 (-0.4)'
    assert df['precision'][2] == '0.3 (-0.355)'
    assert df['sensitivity'][1] == '0.2 (0.0)'
    assert df['f1_score'][2] == '0.3 (-0.45)'
    assert df['number_of_samples'][1] == 2000
    assert df['number_of_samples'][2] == 2000
    assert df['number_of_classes'][1] == 2
    assert df['number_of_classes'][2] == 2

    dataframes = utils.load_csv_to_dataframe(test_individual_results_csv_paths)
    df = utils.results_differential(dataframes, mode='individual')

    assert len(df) == 6

    sensitivity_values_expected = [0.9, '0.2 (-0.7)', '0.15 (-0.75)', 0.1, '0.1 (0.0)', '0.25 (+0.15)']
    for i, val in enumerate(df['sensitivity']):
        assert val == sensitivity_values_expected[i]

    precision_values_expected = [0.55, '0.4 (-0.15)', '0.35 (-0.2)', 0.2, '0.3 (+0.1)', '0.65 (+0.45)']
    for i, val in enumerate(df['precision']):
        assert val == precision_values_expected[i]

    percentage_samples_values_expected = [50.0, 50.0, 50.0, 50.0, 50.0, 50.0]
    for i, val in enumerate(df['% of samples']):
        assert val == percentage_samples_values_expected[i]


def test_compute_differential_str():
    assert utils.compute_differential_str(0.90, 0.65, 4) == ' (-0.25)'
    assert utils.compute_differential_str(0.65, 0.90, 4) == ' (+0.25)'
