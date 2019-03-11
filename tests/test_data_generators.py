import pytest

from keras_eval.data_generators import AugmentedImageDataGenerator
from keras_model_specs import ModelSpec


def test_augmented_image_data_generator_wrong_scale_size(test_catdog_dataset_path):
    model_spec = ModelSpec.get('test', preprocess_func='mean_subtraction',
                               preprocess_args=[141., 130., 123.], target_size=(224, 224, 3))
    with pytest.raises(ValueError) as exception:
        test_data_generator = AugmentedImageDataGenerator(preprocessing_function=model_spec.preprocess_input,
                                                          data_augmentation={'scale_sizes': 'asd'})
        test_data_generator.flow_from_directory(
            directory=test_catdog_dataset_path,
            batch_size=1,
            target_size=model_spec.target_size[:2],
            class_mode='categorical',
            shuffle=False)
    expected = 'Incorrect format for `scale_sizes`, list of ints or `= default` is expected'
    actual = str(exception).split('ValueError: ')[1]
    assert actual == expected


def test_augmented_image_data_generator_wrong_transforms(test_catdog_dataset_path):
    model_spec = ModelSpec.get('test', preprocess_func='mean_subtraction',
                               preprocess_args=[141., 130., 123.], target_size=(224, 224, 3))

    with pytest.raises(ValueError) as exception:
        test_data_generator = AugmentedImageDataGenerator(preprocessing_function=model_spec.preprocess_input,
                                                          data_augmentation={'scale_sizes': [256],
                                                                             'transforms': ['failure']})
        datagen = test_data_generator.flow_from_directory(
            directory=test_catdog_dataset_path,
            batch_size=1,
            target_size=model_spec.target_size[:2],
            class_mode='categorical',
            shuffle=False)
        datagen.next()

    expected = 'Wrong transform: failure . Check documentation to see the supported ones'
    actual = str(exception).split('ValueError: ')[1]
    assert actual == expected

    with pytest.raises(ValueError) as exception:
        test_data_generator = AugmentedImageDataGenerator(preprocessing_function=model_spec.preprocess_input,
                                                          data_augmentation={'scale_sizes': [256],
                                                                             'transforms': 'blah'})
        datagen = test_data_generator.flow_from_directory(
            directory=test_catdog_dataset_path,
            batch_size=1,
            target_size=model_spec.target_size[:2],
            class_mode='categorical',
            shuffle=False)

        datagen.next()

    expected = 'Incorrect format for `transforms`, a list of transforms is expected'
    actual = str(exception).split('ValueError: ')[1]
    assert actual == expected


def test_augmented_image_data_generator_wrong_crop_original(test_catdog_dataset_path):
    model_spec = ModelSpec.get('test', preprocess_func='mean_subtraction',
                               preprocess_args=[141., 130., 123.], target_size=(224, 224, 3))

    with pytest.raises(ValueError) as exception:
        test_data_generator = AugmentedImageDataGenerator(preprocessing_function=model_spec.preprocess_input,
                                                          data_augmentation={'crop_original': 'fail',
                                                                             'scale_sizes': [256]})
        datagen = test_data_generator.flow_from_directory(
            directory=test_catdog_dataset_path,
            batch_size=1,
            target_size=model_spec.target_size[:2],
            class_mode='categorical',
            shuffle=False)

        datagen.next()

    expected = 'crop_original mode entered not supported, only `center_crop` is being supported now'
    actual = str(exception).split('ValueError: ')[1]
    assert actual == expected


def test_augmented_image_data_generator_wrong_arguments(test_catdog_dataset_path):
    model_spec = ModelSpec.get('test', preprocess_func='mean_subtraction',
                               preprocess_args=[141., 130., 123.], target_size=(224, 224, 3))

    with pytest.raises(ValueError) as exception:
        test_data_generator = AugmentedImageDataGenerator(preprocessing_function=model_spec.preprocess_input,
                                                          data_augmentation={'error': 123})
        datagen = test_data_generator.flow_from_directory(
            directory=test_catdog_dataset_path,
            batch_size=1,
            target_size=model_spec.target_size[:2],
            class_mode='categorical',
            shuffle=False)

        datagen.next()

    expected = 'data_augmentation dictionary should contain `crop_original`, `scale_sizes` or `transforms` as keys'
    actual = str(exception).split('ValueError: ')[1]
    assert actual == expected

    model_spec = ModelSpec.get('test', preprocess_func='mean_subtraction',
                               preprocess_args=[141., 130., 123.], target_size=(224, 224, 3))

    with pytest.raises(ValueError) as exception:
        test_data_generator = AugmentedImageDataGenerator(preprocessing_function=model_spec.preprocess_input,
                                                          data_augmentation=[1, 2, 3])
        datagen = test_data_generator.flow_from_directory(
            directory=test_catdog_dataset_path,
            batch_size=1,
            target_size=model_spec.target_size[:2],
            class_mode='categorical',
            shuffle=False)

        datagen.next()

    expected = '`data_augmentation` is a %s and it should be a dictionary' % type([1, 2, 3])
    actual = str(exception).split('ValueError: ')[1]
    assert actual == expected


def test_augmented_image_data_generator(test_catdog_dataset_path):
    model_spec = ModelSpec.get('test', preprocess_func='mean_subtraction',
                               preprocess_args=[141., 130., 123.], target_size=(224, 224, 3))
    test_data_generator = AugmentedImageDataGenerator(preprocessing_function=model_spec.preprocess_input,
                                                      data_augmentation={'scale_sizes': 'default',
                                                                         'transforms': ['horizontal_flip']})

    datagen = test_data_generator.flow_from_directory(
        directory=test_catdog_dataset_path,
        batch_size=1,
        target_size=model_spec.target_size[:2],
        class_mode='categorical',
        shuffle=False)

    batch_x, batch_y = datagen.next()
    assert batch_x.shape == (1, 144, 224, 224, 3)
    assert len(batch_y) == 144

    test_data_generator = AugmentedImageDataGenerator(preprocessing_function=model_spec.preprocess_input,
                                                      data_augmentation={'scale_sizes': [256],
                                                                         'transforms': ['horizontal_flip',
                                                                                        'vertical_flip',
                                                                                        'rotate_90',
                                                                                        'rotate_180',
                                                                                        'rotate_270']})

    datagen = test_data_generator.flow_from_directory(
        directory=test_catdog_dataset_path,
        batch_size=1,
        target_size=model_spec.target_size[:2],
        class_mode='categorical',
        shuffle=False)

    batch_x, batch_y = datagen.next()
    assert batch_x.shape == (1, 108, 224, 224, 3)
    assert len(batch_y) == 108

    test_data_generator = AugmentedImageDataGenerator(preprocessing_function=model_spec.preprocess_input,
                                                      data_augmentation={'transforms': ['horizontal_flip',
                                                                                        'vertical_flip',
                                                                                        'rotate_90',
                                                                                        'rotate_180',
                                                                                        'rotate_270']})

    datagen = test_data_generator.flow_from_directory(
        directory=test_catdog_dataset_path,
        batch_size=1,
        target_size=model_spec.target_size[:2],
        class_mode='categorical',
        shuffle=False)

    batch_x, batch_y = datagen.next()
    assert batch_x.shape == (1, 6, 224, 224, 3)
    assert len(batch_y) == 6

    test_data_generator = AugmentedImageDataGenerator(preprocessing_function=model_spec.preprocess_input,
                                                      data_augmentation={'scale_sizes': [256],
                                                                         'crop_original': 'center_crop',
                                                                         'transforms': ['horizontal_flip',
                                                                                        'vertical_flip',
                                                                                        'rotate_90',
                                                                                        'rotate_180',
                                                                                        'rotate_270']})

    datagen = test_data_generator.flow_from_directory(
        directory=test_catdog_dataset_path,
        batch_size=1,
        target_size=model_spec.target_size[:2],
        class_mode='categorical',
        shuffle=False)

    batch_x, batch_y = datagen.next()
    assert batch_x.shape == (1, 108, 224, 224, 3)
    assert len(batch_y) == 108
