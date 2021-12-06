from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path
import os

def prep_fn(img):
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) * 2
    return img


class DataLoader():

    def __init__(self, data_dir = "../data/combined", split = 0.2, batch_size = 64, task = 'full'):
        self.data_dir = data_dir
        self.split = split
        self.batch_size = batch_size
        self.task = task

        self.data_generator = ImageDataGenerator(preprocessing_function=prep_fn,
                                            horizontal_flip=True,
                                            vertical_flip=True)
        if task == 'planet':
                    
            self.train_generator = self.data_generator.flow_from_directory(
                os.path.join(data_dir, "train"),  # This is the source directory for training images
                classes = ['combinedPlanetForests', 'combinedPlanetOrchards'],#['planetSplitForests/train', 'planetSplitOrchards/train'],
                target_size=(224, 224),
                batch_size=self.batch_size,
                # Use binary labels
                class_mode='binary')

            self.val_generator = self.data_generator.flow_from_directory(
                os.path.join(data_dir, "val"),  # This is the source directory for training images
                classes = ['combinedPlanetForests', 'combinedPlanetOrchards'],#['planetSplitForests/val', 'planetSplitOrchards/val'],
                target_size=(224, 224),
                batch_size=self.batch_size,
                shuffle=False,
                # Use binary labels
                class_mode='binary')

        elif task == 'planet-small':
             
            self.data_generator = ImageDataGenerator(rescale=1/255,
                                            horizontal_flip=True,
                                            vertical_flip=True, validation_split = split)       
            self.train_generator = self.data_generator.flow_from_directory(
                os.path.join(data_dir),#, "data2"),#, "train"),  # This is the source directory for training images
                classes = ['planetSinglesForests', 'planetSinglesOrchards'],#['planetSplitForests/train', 'planetSplitOrchards/train'],
                target_size=(224, 224),
                batch_size=self.batch_size,
                # Use binary labels
                class_mode='binary',
                subset='training')

            self.val_generator = self.data_generator.flow_from_directory(
                os.path.join(data_dir),#, "data2"),#, "val"),  # This is the source directory for training images
                classes = ['planetSinglesForests', 'planetSinglesOrchards'],#['planetSplitForests/val', 'planetSplitOrchards/val'],
                target_size=(224, 224),
                batch_size=self.batch_size,
                shuffle=False,
                subset='validation',
                # Use binary labels
                class_mode='binary')


        else:
            classes = ['combinedGoogleMapsForests', 'combinedGoogleMapsOrchards']
            self.train_generator = self.data_generator.flow_from_directory(
                os.path.join(data_dir, "train"),  # This is the source directory for training images
                classes = classes,
                target_size=(224, 224),
                batch_size=self.batch_size,
                # Use binary labels
                class_mode='binary')

            self.val_generator = self.data_generator.flow_from_directory(
                os.path.join(data_dir, "val"),  # This is the source directory for training images
                classes = classes,
                target_size=(224, 224),
                batch_size=self.batch_size,
                shuffle=False,
                # Use binary labels
                class_mode='binary')


        def make_generator(path, class1, class2, shuffle):
            return self.data_generator.flow_from_directory(
                path,
                classes = [class1,class2],
                target_size=(224,224),
                batch_size=self.batch_size,
                shuffle=shuffle,
                class_mode='binary')

        self.multi_train_google = make_generator("../data/GoogleMapsNewImagery/train/"," imagesGoogleMapsForestsGreater2Hect"," imagesGoogleMapsOrchardsGreater2Hect/", True)
        self.multi_val_google = make_generator("../data/GoogleMapsNewImagery/val/","imagesGoogleMapsForestsGreater2Hect", "imagesGoogleMapsOrchardsGreater2Hect", False)
        self.multi_train_planet = make_generator("../planet/train/","combinedPlanetForests", "combinedPlanetOrchards", True)
        self.multi_val_planet = make_generator("../planet/val/","combinedPlanetForests", "combinedPlanetOrchards", False)
        #self.multi_train_planet = make_generator("../../angelats11_gmail_com/planetSplit/train/","planetImageryForestsGreater2Hect", "planetImageryOrchardsGreater2Hect", True)
        #self.multi_val_planet = make_generator("../../angelats11_gmail_com/planetSplit/val/","planetImageryForestsGreater2Hect", "planetImageryOrchardsGreater2Hect", False)
    #Used this when I was playing around with featurewise_center=True in ImageDataGenerator
    def fit(self):
        self.data_generator.fit(load_all_images(
            os.path.join(self.data_dir, "ImagesGoogleMapsForests"), 224, 224))

    def generate_multiple(self, gen1, gen2):
        while True:
            data1 = gen1.next()
            data2 = gen2.next()
            yield [data1[0], data2[0]], data1[1]

    def separate_data(self, data_generator):
        data_list = []
        labels_list = []
        batch_index = 0

        while batch_index <= data_generator.batch_index:
            data = data_generator.next()
            data_list.append(data[0])
            labels_list.append(data[1])
            batch_index = batch_index + 1

        # now, data_array is the numeric data of whole images
        data_array = np.asarray(data_list)
        labels_array = np.asarray(labels_list)

        return data_array, labels_array


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


