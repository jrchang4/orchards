import numpy as np
import tensorflow as tf
#from tensorflow import keras.preprocessing.image.Iterator
    
class Iterator(tf.keras.preprocessing.image.Iterator):
    def read_image(path, rescale=None):
            #key="{},{}".format(path,rescale)
            #if key in read_image_cache:
            #    return read_image_cache[key]
            #else:
        with rasterio.open(path) as img:
            data=img.read()
            data=np.moveaxis(data,0,-1)
            if rescale!=None:
                data=data*rescale
                    #read_image_cache[key]=data
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


    def __init__(self, n, batch_size, shuffle, seed):
        super().__init__(self, n, batch_size, shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        """Gets a batch of transformed samples.

        # Arguments
            index_array: Array of sample indices to include in batch.

        # Returns
            A batch of transformed samples.
        """
        batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=self.dtype)
        # build batch of image data
        # self.filepaths is dynamic, is better to call it once outside the loop
        filepaths = self.filepaths
        for i, j in enumerate(index_array):
            img = load_img(filepaths[j],
                           color_mode=self.color_mode,
                           target_size=self.target_size,
                           interpolation=self.interpolation)
            x = img_to_array(img, data_format=self.data_format)
            # Pillow images should be closed after `load_img`,
            # but not PIL images.
            if hasattr(img, 'close'):
                img.close()
            if self.image_data_generator:
                params = self.image_data_generator.get_random_transform(x.shape)
                x = self.image_data_generator.apply_transform(x, params)
                x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(
                    prefix=self.save_prefix,
                    index=j,
                    hash=np.random.randint(1e7),
                    format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        # build batch of labels
        if self.class_mode == 'input':
            batch_y = batch_x.copy()
        elif self.class_mode in {'binary', 'sparse'}:
            batch_y = np.empty(len(batch_x), dtype=self.dtype)
            for i, n_observation in enumerate(index_array):
                batch_y[i] = self.classes[n_observation]
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), len(self.class_indices)),
                               dtype=self.dtype)
            for i, n_observation in enumerate(index_array):
                batch_y[i, self.classes[n_observation]] = 1.
        elif self.class_mode == 'multi_output':
            batch_y = [output[index_array] for output in self.labels]
        elif self.class_mode == 'raw':
            batch_y = self.labels[index_array]
        else:
            return batch_x
        if self.sample_weight is None:
            return batch_x, batch_y
        else:
            return batch_x, batch_y, self.sample_weight[index_array]


