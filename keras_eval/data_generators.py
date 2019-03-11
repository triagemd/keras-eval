import os
import numpy as np
import PIL

from keras_preprocessing.image import ImageDataGenerator, DirectoryIterator, array_to_img, load_img, img_to_array
from keras_preprocessing import get_keras_submodule

backend = get_keras_submodule('backend')


class AugmentedDirectoryIterator(DirectoryIterator):
    '''

    AugmentedDirectoryIterator inherits from DirectoryIterator:
    (https://github.com/keras-team/keras-preprocessing/blob/master/keras_preprocessing/image.py#L1811)

    This implementation adds the functionality of computing multiple crops following the work Going Deeper with
    Convolutions (https://arxiv.org/pdf/1409.4842.pdf) and allowing the use of transforms on such crops.

    It includes the addition of data_augmentation as an argument. It is a dictionary consisting of 3 elements:

    - 'scale_sizes': 'default' (4 similar scales to Original paper) or a list of sizes. Each scaled image then
    will be cropped into three square parts. For each square, we then take the 4 corners and the center "target_size"
    crop as well as the square resized to "target_size".
    - 'transforms': list of transforms to apply to these crops in addition to not
    applying any transform ('horizontal_flip', 'vertical_flip', 'rotate_90', 'rotate_180', 'rotate_270' are
    supported now).
    - 'crop_original': 'center_crop' mode allows to center crop the original image prior do the rest of transforms,
    scalings + croppings.

    If 'scale_sizes' is None the image will be resized to "target_size" and transforms will be applied over that image.

    For instance: data_augmentation={'scale_sizes':'default', 'transforms':['horizontal_flip', 'rotate_180'],
    'crop_original':'center_crop'}

    For 144 crops as GoogleNet paper, select data_augmentation={'scale_sizes':'default',
    'transforms':['horizontal_flip']}
    This results in 4x3x6x2 = 144 crops per image.

    '''

    def __init__(self, directory, image_data_generator, data_augmentation,
                 target_size=(299, 299), color_mode='rgb',
                 classes=None, class_mode='categorical',
                 batch_size=32, shuffle=True, seed=None,
                 data_format='channels_last',
                 save_to_dir=None, save_prefix='', save_format='png',
                 follow_links=False,
                 subset=None,
                 interpolation='nearest',
                 ):

        # Solve Reload Errors
        self.as_super = super(AugmentedDirectoryIterator, self)
        self.as_super.__init__(directory,
                               image_data_generator,
                               target_size,
                               color_mode,
                               classes,
                               class_mode,
                               batch_size,
                               shuffle,
                               seed,
                               data_format,
                               save_to_dir,
                               save_prefix,
                               save_format,
                               follow_links,
                               subset,
                               interpolation,
                               )

        self._check_data_augmentation_keys(data_augmentation)
        self.data_augmentation = data_augmentation

        self.crop_original = None
        if 'crop_original' in data_augmentation.keys():
            self.crop_original = self.data_augmentation['crop_original']

        self.transforms = ['none']
        if 'transforms' in self.data_augmentation.keys():
            if isinstance(self.data_augmentation['transforms'], list):
                self.transforms += self.data_augmentation['transforms']
            else:
                raise ValueError('Incorrect format for `transforms`, a list of transforms is expected')
        self.n_transforms = len(self.transforms)

        if 'scale_sizes' in self.data_augmentation.keys():
            if self.data_augmentation['scale_sizes'] == 'default':
                self.scale_sizes = self._get_default_sizes(self.target_size[0])
            elif isinstance(self.data_augmentation['scale_sizes'], list) and \
                    all(isinstance(x, int) for x in self.data_augmentation['scale_sizes']):
                self.scale_sizes = []
                for size in self.data_augmentation['scale_sizes']:
                    size = round(size)
                    # Use sizes that are multiples of 2
                    if size % 2 != 0:
                        size += 1
                    self.scale_sizes.append(size)
            else:
                raise ValueError('Incorrect format for `scale_sizes`, list of ints or `= default` is expected')

            self.n_crops = len(self.scale_sizes) * 6 * 3 * self.n_transforms
        else:
            self.scale_sizes = None
            self.n_crops = self.n_transforms

    @staticmethod
    def _check_data_augmentation_keys(data_augmentation):
        if isinstance(data_augmentation, dict):
            keys = data_augmentation.keys()
            if 'scale_sizes' not in keys and 'transforms' not in keys and 'crop_original' not in keys:
                raise ValueError('data_augmentation dictionary should contain '
                                 '`crop_original`, `scale_sizes` or `transforms` as keys')
        else:
            raise ValueError('`data_augmentation` is a %s and it should be a dictionary' % type(data_augmentation))

    @staticmethod
    def _get_default_sizes(target_size, multipliers=(1.1, 1.2, 1.3, 1.4)):
        sizes = []
        for multiplier in multipliers:
            size = round(target_size * multiplier)
            if size % 2 != 0:
                size += 1
            sizes.append(size)
        return sizes

    @staticmethod
    def _get_3_crops(image):
        '''

        Args:
            image: PIL Image

        Returns: 3 square sized crops of the image. Top, central and bottom in the case of a vertical image
        and left, central and right in the case of a horizontal image.

        '''

        w, h = image.size
        w_center = w / 2
        h_center = h / 2

        if w >= h:
            im_1 = image.crop((0, 0, h, h))
            im_2 = image.crop((w_center - h / 2, 0, w_center + h / 2, h))
            im_3 = image.crop((w - h, 0, w, h))
        else:
            im_1 = image.crop((0, 0, w, w))
            im_2 = image.crop((0, h_center - w / 2, w, h_center + w / 2))
            im_3 = image.crop((0, h - w, w, h))

        return [im_1, im_2, im_3]

    @staticmethod
    def _apply_transform(image, transform):
        '''

        Args:
            image: PIL input image
            transform: Transform to apply

        Returns: Transformed image in PIL format.

        '''
        transform_dict = {'horizontal_flip': PIL.Image.FLIP_LEFT_RIGHT, 'vertical_flip': PIL.Image.FLIP_TOP_BOTTOM,
                          'rotate_90': PIL.Image.ROTATE_90, 'rotate_180': PIL.Image.ROTATE_180,
                          'rotate_270': PIL.Image.ROTATE_270}
        if transform == 'none':
            return image
        elif transform in transform_dict.keys():
            return image.transpose(transform_dict[transform])
        else:
            raise ValueError('Wrong transform: %s . Check documentation to see the supported ones' % transform)

    def _apply_augmentation(self, image, size, transforms):
        '''

        Args:
            image: PIL input image
            size: Target output size
            transforms: List of transforms to apply

        Returns: Crops plus transformations done to the input image

        '''
        crops = []

        target_w, target_h = size
        images_cropped_at_scale = self._get_3_crops(image)

        for img in images_cropped_at_scale:
            w, h = img.size
            w_center = w / 2
            h_center = h / 2

            for transform in transforms:
                # Central Crop
                crops.append(self._apply_transform(img.crop((w_center - target_w / 2,
                                                             h_center - target_h / 2,
                                                             w_center + target_w / 2,
                                                             h_center + target_h / 2))
                                                   .resize((target_w, target_h)), transform))
                # Left-Up
                crops.append(self._apply_transform(img.crop((0, 0, target_w, target_h)), transform))
                # Left-Bottom
                crops.append(self._apply_transform(img.crop((0, h - target_h, target_w, h)), transform))
                # Right-Up
                crops.append(self._apply_transform(img.crop((w - target_w, 0, w, target_h)), transform))
                # Right-Bottom
                crops.append(self._apply_transform(img.crop((w - target_w, h - target_h, w, h)), transform))
                # Resized Square
                crops.append(self._apply_transform(img.resize((target_w, target_h)), transform))

        return crops

    def _get_batches_of_transformed_samples(self, index_array):
        grayscale = self.color_mode == 'grayscale'
        batch_x = np.zeros((len(index_array), self.n_crops,) + self.image_shape, dtype=backend.floatx())
        # Build batch of image data
        for i, j in enumerate(index_array):
            crops = []
            fname = self.filenames[j]
            image = load_img(os.path.join(self.directory, fname),
                             grayscale=grayscale,
                             target_size=None,
                             interpolation=self.interpolation)

            if self.crop_original == 'center_crop':
                w, h = image.size
                if w > h:
                    image = image.crop((w / 2 - h / 2, 0, w / 2 + h / 2, h))
                else:
                    image = image.crop((0, h / 2 - w / 2, w, h / 2 + w / 2))
            elif self.crop_original:
                raise ValueError('crop_original mode entered not supported, only `center_crop` is being supported now')

            image_w, image_h = image.size

            if self.scale_sizes is not None:
                for size in self.scale_sizes:
                    if image_w <= image_h:
                        img = image.resize((size, round(image_h / image_w * size)))
                    else:
                        img = image.resize((round(image_w / image_h * size), size))
                    crops += self._apply_augmentation(img, self.target_size, self.transforms)
            else:
                crops += [self._apply_transform(image.resize(self.target_size), transform)
                          for transform in self.transforms]

            for c_i, crop in enumerate(crops):
                x = img_to_array(crop, data_format=self.data_format)
                x = self.image_data_generator.standardize(x)
                batch_x[i, c_i] = x

        # Optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i, j in enumerate(index_array):
                for crop in batch_x[i]:
                    img = array_to_img(crop, self.data_format, scale=True)
                    fname = '{prefix}_{index}_{hash}.{format}'.format(
                        prefix=self.save_prefix,
                        index=j,
                        hash=np.random.randint(1e7),
                        format=self.save_format)
                    img.save(os.path.join(self.save_to_dir, fname))

        # Build batch of labels
        if self.class_mode == 'input':
            batch_y = batch_x.copy()
        elif self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype(backend.floatx())
        elif self.class_mode == 'categorical':
            batch_y = np.zeros(
                (len(batch_x), self.num_classes),
                dtype=backend.floatx())
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        else:
            return batch_x
        return batch_x, [batch_y for n in range(self.n_crops)]


class AugmentedImageDataGenerator(ImageDataGenerator):
    '''

    ImageDataGenerator that returns multiple outputs using AugmentedDirectoryIterator as iterator. It has the addition
    of `data_augmentation` argument to control the type of modifications desired.

    '''

    def __init__(self,
                 data_augmentation,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 zca_epsilon=1e-6,
                 rotation_range=0.,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 brightness_range=None,
                 shear_range=0.,
                 zoom_range=0.,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,
                 preprocessing_function=None,
                 data_format='channels_last',
                 validation_split=0.0,
                 ):
        self.as_super = super(AugmentedImageDataGenerator, self)
        self.as_super.__init__(featurewise_center=featurewise_center,
                               samplewise_center=samplewise_center,
                               featurewise_std_normalization=featurewise_std_normalization,
                               samplewise_std_normalization=samplewise_std_normalization,
                               zca_whitening=zca_whitening,
                               zca_epsilon=zca_epsilon,
                               rotation_range=rotation_range,
                               width_shift_range=width_shift_range,
                               height_shift_range=height_shift_range,
                               brightness_range=brightness_range,
                               shear_range=shear_range,
                               zoom_range=zoom_range,
                               channel_shift_range=channel_shift_range,
                               fill_mode=fill_mode,
                               cval=cval,
                               horizontal_flip=horizontal_flip,
                               vertical_flip=vertical_flip,
                               rescale=rescale,
                               preprocessing_function=preprocessing_function,
                               data_format=data_format,
                               validation_split=validation_split,
                               )

        self.data_augmentation = data_augmentation

    def flow_from_directory(self, directory,
                            target_size=(256, 256), color_mode='rgb',
                            classes=None, class_mode='categorical',
                            batch_size=32, shuffle=True, seed=None,
                            save_to_dir=None,
                            save_prefix='',
                            save_format='png',
                            follow_links=False,
                            subset=None,
                            interpolation='nearest',
                            ):
        return AugmentedDirectoryIterator(
            directory, self,
            data_augmentation=self.data_augmentation,
            target_size=target_size, color_mode=color_mode,
            classes=classes, class_mode=class_mode,
            data_format=self.data_format,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            follow_links=follow_links,
            subset=subset,
            interpolation=interpolation,
        )
