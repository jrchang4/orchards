from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path
import os

class DataLoader():
    def __init__(self, data_dir = "../data/", split = 0.2, batch_size = 64, task = 'full'):
        self.data_dir = data_dir
        self.split = split
        self.batch_size = batch_size
        self.task = task

        self.data_generator = ImageDataGenerator(rescale=1/255,
                                            horizontal_flip=True,
                                            vertical_flip=True)

        task_class = 'contrast_eq_OilPalm' if task == 'palm' else 'contrast_eq_orchards'
        
        self.train_generator = self.data_generator.flow_from_directory(
            os.path.join(data_dir, "data2", "train"),  # This is the source directory for training images
            classes = ['contrast_eq_forests', task_class],
            target_size=(224, 224),
            batch_size=self.batch_size,
            # Use binary labels
            class_mode='binary')

        self.val_generator = self.data_generator.flow_from_directory(
            os.path.join(data_dir, "data2", "val"),  # This is the source directory for training images
            classes = ['contrast_eq_forests', task_class],
            target_size=(224, 224),
            batch_size=self.batch_size,
            shuffle=False,
            # Use binary labels
            class_mode='binary')

    #Used this when I was playing around with featurewise_center=True in ImageDataGenerator
    def fit(self):
        self.data_generator.fit(load_all_images(
            os.path.join(self.data_dir, "ImagesGoogleMapsForests"), 224, 224))


def read_pil_image(img_path, height, width):
    with open(img_path, 'rb') as f:
        return np.array(Image.open(f).convert('RGB').resize((width, height)))

def load_all_images(dataset_path, height, width, img_ext='jpg'):
    return np.array([read_pil_image(str(p), height, width) for p in
                     Path(dataset_path).rglob("*." + img_ext)])



class TifDataGenerator(tf.keras.preprocessing.image.ImageDataGenerator):
        def __init__(self,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 zca_epsilon=1e-6,
                 rotation_range=0,
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
                 validation_split=0.2,
                 dtype='float32'):
                super().__init__(featurewise_center,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 zca_epsilon=1e-6,
                 rotation_range=0,
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
                 validation_split=0.2,
                 dtype='float32')

        def read_image(path, rescale=None):
                key="{},{}".format(path,rescale)
                if key in read_image_cache:
                    return read_image_cache[key]
                else:
                    with rasterio.open(path) as img:
                        data=img.read()
                        data=np.moveaxis(data,0,-1)
                        if rescale!=None:
                            data=data*rescale
                        read_image_cache[key]=data
                return data


        def load_img(path, time_series=True, target_size=None,
             interpolation='nearest'):

            ### Work on custom loader for 4-channel (or higher) data
            """Loads an image into PIL format.

            # Arguments
                path: Path to image file.
                grayscale: DEPRECATED use `color_mode="grayscale"`.
                color_mode: The desired image format. One of "grayscale", "rgb", "rgba".
                    "grayscale" supports 8-bit images and 32-bit signed integer images.
                    Default: "rgb".
                target_size: Either `None` (default to original size)
                    or tuple of ints `(img_height, img_width)`.
                interpolation: Interpolation method used to resample the image if the
                    target size is different from that of the loaded image.
                    Supported methods are "nearest", "bilinear", and "bicubic".
                    If PIL version 1.1.3 or newer is installed, "lanczos" is also
                    supported. If PIL version 3.4.0 or newer is installed, "box" and
                    "hamming" are also supported.
                    Default: "nearest".
            # Returns
                A PIL Image instance.
            # Raises
                ImportError: if PIL is not available.
                ValueError: if interpolation method is not supported.
            """
            if pil_image is None:
                raise ImportError('Could not import PIL.Image. '
                                  'The use of `load_img` requires PIL.')
            with open(path, 'rb') as f:

                img = read_image(f)
                if target_size is not None:
                    width_height_tuple = (target_size[1], target_size[0])
                    if img.size != width_height_tuple:
                        if interpolation not in _PIL_INTERPOLATION_METHODS:
                            raise ValueError(
                                'Invalid interpolation method {} specified. Supported '
                                'methods are {}'.format(
                                    interpolation,
                                    ", ".join(_PIL_INTERPOLATION_METHODS.keys())))
                        resample = _PIL_INTERPOLATION_METHODS[interpolation]
                        img = img.resize(width_height_tuple, resample)
            return img


