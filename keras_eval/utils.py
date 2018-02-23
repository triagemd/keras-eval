import os
import keras
import json


def load_multi_model(models_path, custom_objects=None):
    '''
    Loads multiple models stored in `models_path`.

    Args:
       models_path: a string indicating the directory were models are stored.
       custom_objects: dict mapping class names (or function names) of custom (non-Keras) objects to class/functions.
                    e.g. for mobilenet models: {'relu6': mobilenet.relu6, 'DepthwiseConv2D': mobilenet.DepthwiseConv2D}

    Returns: Models list

    '''

    models = []
    model_specs = []
    num_models = 0
    model_extensions = ['.h5', '.hdf5']
    for dirpath, dirnames, models in os.walk(models_path):
        for model in models:
            if model.endswith(tuple(model_extensions)):
                print('Loading model ', model)
                model, model_spec = load_model(os.path.join(dirpath, model), custom_objects)
                models.append(model)
                model_specs.append(model_spec)
                num_models += 1
            else:
                raise ValueError('Model files must be either h5 or hdf5 files. Found : ' + str(
                    os.path.splitext(model)[1] + 'file.'))

    print('Models loaded: ', num_models)
    return models


def load_model(model_path, specs_path=None, custom_objects=None):
    model = keras.models.load_model(model_path, custom_objects)
    if specs_path is None:
        split_path = model_path.split('/')
        specs_path = split_path[:-1]
        specs_name = split_path[-1].replace('.h5', '_model_spec.json')
    with open(os.path.join(specs_path, specs_name), 'r') as f:
        model_spec = json.load(f)
    return model, model_spec


def create_image_generator(data_dir, batch_size, target_size, preprocessing_function=None):
    '''
    Creates a Keras image generator
    Args:
        batch_size: N images per batch
        preprocessing_function: Function to preprocess the images
        target_size: Size of the images

    Returns: Keras generator without shuffling samples and ground truth labels associated with generator

    '''
    test_gen = image.ImageDataGenerator(preprocessing_function=preprocessing_function)

    generator = test_gen.flow_from_directory(data_dir, batch_size=batch_size, target_size=target_size,
                                             class_mode='categorical', shuffle=False)

    gt_labels = keras.utils.np_utils.to_categorical(generator.classes, generator.num_classes)

    return generator, gt_labels


def load_preprocess_image(img_name, preprocess_func, target_size):
    """
    Return a preprocessed images (probably to use within a deep neural net).

    Args:
        img_name: a string indicating the name and path of the image.
        preprocess_func: a preprocessing function to apply to the image.
        target_size: size to resize the image to.

    Returns: the preprocessed image.

    """

    # Load the image.
    img = image.load_img(img_name, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    # Pre-process the image.
    if preprocess_func is not None:
        x = preprocess_func(x)
    return x


def load_preprocess_images(img_paths, preprocess_func, target_size):
    """
    Return an array of preprocessed images.

    Args:
        img_paths: a list of paths to images.
        preprocess_func: a preprocessing function to apply to each image.
        target_size: size the image should be resized to.

    Returns:
        pre_imgs: an array of preprocessed images.

    """
    pre_imgs = []

    for img_path in img_paths:
        pre_img = load_preprocess_image(img_path, preprocess_func=preprocess_func, target_size=target_size)
        pre_imgs.append(np.squeeze(pre_img))

    pre_imgs = np.asarray(pre_imgs)
    return pre_imgs


def combine_ensemble_probs(probs, combination_mode=None):
    '''
    Args:
        probs: Probailities given by the ensemble of models
        combination_mode: combination_mode: 'arithmetic' / 'geometric' / 'harmonic' mean of the predictions or 'maximum'
           probability value

    Returns: Probabilities combined
    '''
    # Probabilities of the ensemble input=[n_models, n_images, n_class] --> output=[n_images, n_class]

    # Join probabilities given by an ensemble of models following combination mode

    combiners = {
        'arithmetic': np.mean,
        'geometric': scipy.stats.gmean,
        'harmonic': scipy.stats.hmean,
        'maximum': np.amax
    }
    if combination_mode is None:
        raise ValueError('combination_mode is required')
    elif combination_mode not in combiners.keys():
        raise ValueError('Error: invalid option for `combination_mode` ' + str(combination_mode))
    combiner = combiners[combination_mode]
    return combiner(probs, axis=0)


# Custom preprocessing: dataset mean subtraction
def preprocessing_mean_dataset_subtraction(x):
    # Should be centered in 0 --> Between [-1,1]
    x -= mean_dataset
    x /= 255.
    x *= 2.
    return x


# Custom preprocessing: dataset mean subtraction
def preprocess_between_plus_minus_1(x):
    # Should be centered in 0 --> Between [-1,1]
    x /= 255.
    x -= 0.5
    x *= 2.
    return x