from keras_eval.data_generators import AugmentedImageDataGenerator


def test_augmented_image_data_generator_wrong_scale_size():
    model_spec = ModelSpec.get('test', preprocess_func='mean_subtraction',
                               preprocess_args=[141., 130., 123.], target_size=(224, 224, 3))
    with pytest.raises(ValueError) as exception:
        test_data_generator = AugmentedImageDataGenerator(preprocessing_function=model_spec.preprocess_input,
                                                          data_augmentation={'scale_sizes': 'asd'})
    expected = 'Incorrect format for `scale_sizes`, list or `default` is expected'
    actual = str(exception).split('ValueError: ')[1]
    assert actual == expected


def test_augmented_image_data_generator_wrong_transforms():
    model_spec = ModelSpec.get('test', preprocess_func='mean_subtraction',
                               preprocess_args=[141., 130., 123.], target_size=(224, 224, 3))
    with pytest.raises(ValueError) as exception:
        test_data_generator = AugmentedImageDataGenerator(preprocessing_function=model_spec.preprocess_input,
                                                          data_augmentation={'scale_sizes': [256],
                                                                             'transforms': ['failure']})
    expected = 'Wrong transform failure check documentation to see the supported ones'
    actual = str(exception).split('ValueError: ')[1]
    assert actual == expected


def test_augmented_image_data_generator(test_catdog_dataset_path):
    model_spec = ModelSpec.get('test', preprocess_func='mean_subtraction',
                               preprocess_args=[141., 130., 123.], target_size=(224, 224, 3))

    test_data_generator = AugmentedImageDataGenerator(preprocessing_function=model_spec.preprocess_input,
                                                      data_augmentation={'scale_sizes': [256],
                                                                         'transforms': ['horizontal_flip',
                                                                                        'vertical_flip',
                                                                                        'rotate_90',
                                                                                        'rotate_180',
                                                                                        'rotate_270']})

    test_data_generator.flow_from_directory(
        directory=test_catdog_dataset_path,
        batch_size=1,
        target_size=model_spec.target_size[:2],
        class_mode='categorical',
        shuffle=False)

    import pdb
    pdb.set_trace()
    for batch_x, batch_y in test_data_generator:
        assert batch_x.shape == [1, 36, 224, 224]
        assert batch_y.shape == [1, 36, 224, 224]
        break
