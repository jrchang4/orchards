from tensorflow.keras.preprocessing.image import ImageDataGenerator
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

        #task_class = 'contrast_eq_OilPalm' if task == 'palm' else 'contrast_eq_orchards'
        combined = True
        if combined:
            classes = ['combinedGoogleMapsForests', 'combinedGoogleMapsOrchards']
            data_dir = "../data/combined"
        else:
            classes = ["ImagesGoogleMapsForests", "ImagesGoogleMapsOrchards"]
            data_dir = "../data/data2"

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



