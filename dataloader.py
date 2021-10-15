from tensorflow.keras.preprocessing.image import ImageDataGenerator

class DataLoader():
	def __init__(self, data_dir = "../data/", split = 0.2):
		self.data_dir = data_dir
		self.split = split

		data_generator = ImageDataGenerator(rescale=1/255, validation_split = self.split)

		self.train_generator = data_generator.flow_from_directory(
        data_dir,  # This is the source directory for training images
        classes = ['ImagesGoogleMapsForests', 'ImagesGoogleMapsOrchards'],
        target_size=(224, 224),  # All images will be resized to 200x200
        batch_size=120,
        # Use binary labels
        class_mode='binary', 
        subset = 'training')

		self.val_generator = data_generator.flow_from_directory(
        data_dir,  # This is the source directory for training images
        classes = ['ImagesGoogleMapsForests', 'ImagesGoogleMapsOrchards'],
        target_size=(224, 224),  # All images will be resized to 200x200
        batch_size=120,
        # Use binary labels
        class_mode='binary', 
        subset = 'validation',
        shuffle = False)


