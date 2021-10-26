from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
class DataLoader():
	def __init__(self, data_dir = "../", split = 0.2): #"~/es262-cloud/"
		self.data_dir = data_dir
		self.split = split

		data_generator = TifDataGenerator()#rescale=1/255, validation_split = self.split)

		self.train_generator = data_generator.flow_from_directory(
        data_dir,  # This is the source directory for training images
        #classes = ['imagesGoogleMapsForests', 'imagesGoogleMapsOrchards'],
        classes = ['planetImageryCentroidForests','planetImageryCentroidOrchards'],
	target_size=(224, 224),  # All images will be resized to 200x200
        batch_size=120,
        # Use binary labels
        class_mode='binary', 
        subset = 'training')

		self.val_generator = data_generator.flow_from_directory(
        data_dir,  # This is the source directory for training images
        #classes = ['imagesGoogleMapsForests', 'imagesGoogleMapsOrchards'],
	classes = ['planetImageryCentroidForests','planetImageryCentroidOrchards'],
        target_size=(224, 224),  # All images will be resized to 200x200
        batch_size=120,
        # Use binary labels
        class_mode='binary', 
        subset = 'validation',
        shuffle = False)


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
      
  #def __init__(self,
       #          preprocessing_function=None,
        #         data_format='channels_last',
         #        validation_split=20.0):
                #self.df = df.copy()
                #self.X_col = X_col
                #self.y_col = y_col
                #self.batch_size = batch_size
         #       self.input_size = input_size
         #       self.shuffle = shuffle            
         #       self.n = len(self.df)
         #       self.n_name = df[y_col['name']].nunique()
         #       self.n_type = df[y_col['type']].nunique()
           
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
                

        #        img = pil_image.open(io.BytesIO(f.read()))
                img = read_image(f)
                #if time_series == True:
                #    continue                    
                #else:
                #    raise ValueError('time_series was specified false')
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


        '''def __get_input(self, path, bbox, target_size):
                xmin, ymin, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']

                image = tf.keras.preprocessing.image.load_tif(path)
                image_arr = tf.keras.preprocessing.image.img_to_array(image)

                image_arr = image_arr[ymin:ymin+h, xmin:xmin+w]
                image_arr = tf.image.resize(image_arr,(target_size[0], target_size[1])).numpy()

                return image_arr/255.
         '''

