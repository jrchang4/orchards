from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image
from pathlib import Path
import os

class DataLoader():
    def __init__(self, data_dir = "../data/", split = 0.2):
        self.data_dir = data_dir
        self.split = split

        self.data_generator = ImageDataGenerator(rescale=1/255,
                                            horizontal_flip=True,
                                            vertical_flip=True)
        self.train_generator = self.data_generator.flow_from_directory(
            os.path.join("../data2", "train"),  # This is the source directory for training images
            classes = ['ImagesGoogleMapsForests', 'ImagesGoogleMapsOrchards'],
            target_size=(224, 224),
            batch_size=64,
            # Use binary labels
            class_mode='binary')

        self.val_generator = self.data_generator.flow_from_directory(
            os.path.join("../data2", "val"),  # This is the source directory for training images
            classes = ['ImagesGoogleMapsForests', 'ImagesGoogleMapsOrchards'],
            target_size=(224, 224),
            batch_size=64,
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




