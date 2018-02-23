import os
import keras


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
    num_models = 0
    model_extensions = ['.h5', '.hdf5']
    for dirpath, dirnames, models in os.walk(models_path):
        for model in models:
            if model.endswith(tuple(model_extensions)):
                print('Loading model ', model)
                read_model = keras.models.load_model(os.path.join(dirpath, model), custom_objects)
                models.append(read_model)
                num_models += 1
            else:
                raise ValueError('Model files must be either h5 or hdf5 files. Found : ' + str(
                    os.path.splitext(model)[1] + 'file.'))

    print('Models loaded: ', num_models)
    return models


def load_model(self, model_name, custom_objects=None):
    self.model_name = model_name
    self.model = keras.models.load_model(self.model_name, custom_objects)


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