from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image
from pathlib import Path
import os
import splitfolders

class DataLoader():
    def __init__(self, data_dir = "../data/", split = 0.2):
        self.data_dir = data_dir
        self.split = split

        # splitfolders.ratio(data_dir,  "../data2", seed=1337, ratio=(1 - self.split, self.split),
        #                    group_prefix=None)  # default values

        self.data_generator = ImageDataGenerator(rescale=1/255,
                                            horizontal_flip=True,
                                            vertical_flip=True)

        """self.train_generator = self.data_generator.flow_from_directory(
            data_dir,  # This is the source directory for training images
            classes = ['fake_negative', 'fake_positive'],
            target_size=(224, 224),  # All images will be resized to 200x200
            batch_size=120,
            # Use binary labels
            class_mode='binary',
            # save_to_dir='/Users/ctoups/Documents/Schoolwork/cs325b/resized',
            subset = 'training')

        self.val_generator = self.data_generator.flow_from_directory(
            data_dir,  # This is the source directory for training images
            classes = ['fake_negative', 'fake_positive'],
            target_size=(224, 224),  # All images will be resized to 200x200
            batch_size=120,
            # Use binary labels
            class_mode='binary',
            subset = 'validation')
            """
        self.train_generator = self.data_generator.flow_from_directory(
            os.path.join("../data/data2", "train"),  # This is the source directory for training images
            classes = ['contrast_eq_forests', 'contrast_eq_OilPalm'],
            target_size=(224, 224),
            batch_size=120,
            # Use binary labels
            class_mode='binary')
            # save_to_dir='/Users/ctoups/Documents/Schoolwork/cs325b/resized',
            #subset = 'training')

        self.val_generator = self.data_generator.flow_from_directory(
            os.path.join("../data/data2", "val"),  # This is the source directory for training images
            classes = ['contrast_eq_forests', 'contrast_eq_OilPalm'],
            target_size=(224, 224),
            batch_size=120,
            shuffle=False,
            # Use binary labels
            class_mode='binary')
            #subset = 'validation')

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




