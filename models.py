import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout 

from keras import backend as K
K.set_image_data_format('channels_first')


fully_connected = Sequential([Flatten(input_shape = (224,224,28)), 
                                    Dense(128, activation=tf.nn.relu), 
                                    Dense(1, activation=tf.nn.sigmoid)])



basic_cnn = Sequential()
basic_cnn.add(Conv2D(32,3,padding="same", activation="relu", input_shape=(224,224,28)))
basic_cnn.add(MaxPool2D((2,2), padding='same'))

basic_cnn.add(Conv2D(32, 3, padding="same", activation="relu"))
basic_cnn.add(MaxPool2D((2,2), padding='same'))

basic_cnn.add(Conv2D(64, 3, padding="same", activation="relu"))
basic_cnn.add(MaxPool2D((2,2), padding='same'))
basic_cnn.add(Dropout(0.4))

basic_cnn.add(Flatten())
basic_cnn.add(Dense(128,activation="relu"))
basic_cnn.add(Dense(1, activation="sigmoid"))

vgg_conv = Sequential()
vgg_conv.add(Conv2D(16, 4, padding="same", activation="relu", input_shape=(224,224,28)))

vgg_conv.add(Conv2D(16, 4, padding="same", activation="relu"))
vgg_conv.add(MaxPool2D((2,2), padding='same'))

vgg_conv.add(Conv2D(32, 4, padding="same", activation="relu"))
vgg_conv.add(MaxPool2D((2,2), padding='same'))

vgg_conv.add(Conv2D(32, 4, padding="same", activation="relu"))
vgg_conv.add(MaxPool2D((2,2), padding='same'))

vgg_conv.add(Conv2D(64, 4, padding="same", activation="relu"))
vgg_conv.add(MaxPool2D((2,2), padding='same'))

vgg_conv.add(Conv2D(64, 4, padding="same", activation="relu"))
vgg_conv.add(MaxPool2D((2,2), padding='same'))

vgg_conv.add(Conv2D(128, 4, padding="same", activation="relu"))
vgg_conv.add(MaxPool2D((2,2), padding='same'))

vgg_conv.add(Conv2D(128, 4, padding="same", activation="relu"))
vgg_conv.add(MaxPool2D((2,2), padding='same'))

vgg_conv.add(Conv2D(128, 4, padding="same", activation="relu"))
vgg_conv.add(MaxPool2D((2,2), padding='same'))

vgg_conv.add(Flatten())
vgg_conv.add(Dense(512, activation='relu'))
vgg_conv.add(Dense(256))
vgg_conv.add(Dense(64))
vgg_conv.add(Dense(1, activation="sigmoid"))

#sat_feat = vgg_conv(planet_imgs)



#csize = np.int(np.floor(sat_psize/32))
#d11 = Linear(csize*2*128,512)
#d12 = Linear(512, 256)
#d13 = Linear(256,64)

