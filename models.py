import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Conv2D , MaxPool2D , Flatten , Dropout, Activation, BatchNormalization
#from tensorflow.python.keras import regularizers

from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
#from tensorflow.keras.layers.merge import concatenate
from keras.layers import Concatenate

from keras import backend as K
K.set_image_data_format('channels_last')

from config import get_args

args = get_args()

from keras import backend as K
K.set_image_data_format('channels_first')


fully_connected = Sequential([Flatten(input_shape = (224,224,28)), 
                                    Dense(128, activation=tf.nn.relu), 
                                    Dense(1, activation=tf.nn.sigmoid)])



basic_cnn = Sequential()
<<<<<<< HEAD
basic_cnn.add(Conv2D(32,3,padding="same", activation="relu", input_shape=(224,224,3)))
=======
basic_cnn.add(Conv2D(32,3,padding="same", activation="relu", input_shape=(224,224,28)))
>>>>>>> c891d9a41be2f4a3fc086c636af20209723d693b
basic_cnn.add(MaxPool2D((2,2), padding='same'))

basic_cnn.add(Conv2D(32, 3, padding="same", activation="relu"))
basic_cnn.add(MaxPool2D((2,2), padding='same'))

basic_cnn.add(Conv2D(64, 3, padding="same", activation="relu"))
basic_cnn.add(MaxPool2D((2,2), padding='same'))
basic_cnn.add(Dropout(0.4))

basic_cnn.add(Flatten())
basic_cnn.add(Dense(128,activation="relu"))
basic_cnn.add(Dense(1, activation="sigmoid"))

<<<<<<< HEAD


vgg_conv = Sequential()
vgg_conv.add(Conv2D(16, 4, padding="same", activation="relu", input_shape=(224,224,3)))
=======
vgg_conv = Sequential()
vgg_conv.add(Conv2D(16, 4, padding="same", activation="relu", input_shape=(224,224,28)))
>>>>>>> c891d9a41be2f4a3fc086c636af20209723d693b

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
<<<<<<< HEAD
vgg_conv.add(Flatten())


#vgg_conv.add(Dense(1, activation="sigmoid"))



  
inceptionv3 = tf.keras.applications.InceptionV3(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

for layer in inceptionv3.layers:
  layer.trainable = False

    
from tensorflow.keras import layers

planet_input = Input(shape=(224,224,3), name='planet_input')
def output_layer_multi(model, planet_features):
  # Flatten the output layer to 1 dimension
  x = Flatten()(model.output)
  # Add a fully connected layer with 1,024 hidden units and ReLU activation
  x = Dense(1024, activation='relu')(x)#, kernel_regularizer=regularizers.l2(args.reg),
  #  bias_regularizer=regularizers.l2(args.reg),
  #  activity_regularizer=regularizers.l2(args.reg))(x)
  # Add a dropout rate of 0.2
  
  #x = Dropout(args.dropout)(x)
  # Add a final sigmoid layer for classification
  
  planet_feat = vgg_conv(planet_features)

  y = layers.concatenate([x, planet_feat])

  merged = Dense(128, activation='relu')(y)
  predictions = Dense(1, activation='sigmoid', name='main_output')(merged)


#  x = Dense(1, activation='sigmoid')(x)
  return x

def output_layer(model):
  # Flatten the output layer to 1 dimension
  x = Flatten()(model.output)
  # Add a fully connected layer with 1,024 hidden units and ReLU activation
  x = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(args.reg),
    bias_regularizer=regularizers.l2(args.reg),
    activity_regularizer=regularizers.l2(args.reg))(x)
  # Add a dropout rate of 0.2
  x = Dropout(args.dropout)(x)
  # Add a final sigmoid layer for classification
  x = Dense(1, activation='sigmoid')(x)
  return x



Multimodal = Model(inceptionv3.input, output_layer_multi(inceptionv3, planet_input))
#[inceptionv3.input, planet_input]
Inception = Model(inceptionv3.input, output_layer(inceptionv3))


xception = tf.keras.applications.Xception(
    include_top=False,
    weights="imagenet",
    input_shape=(224,224, 3),
)

for layer in xception.layers:
  layer.trainable = False
Xception = Model(xception.input, output_layer(xception))



  
resnet = tf.keras.applications.ResNet50(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

for layer in resnet.layers:
  layer.trainable = False
  
    
ResNet = Model(resnet.input, output_layer(resnet))


#csize = np.int(np.floor(sat_psize/32))
#d11 = Linear(csize*2*128,512)
#d12 = Linear(512, 256)
#d13 = Linear(256,64)
=======
vgg_conv.add(Dense(1, activation="sigmoid"))

#sat_feat = vgg_conv(planet_imgs)

>>>>>>> c891d9a41be2f4a3fc086c636af20209723d693b


#csize = np.int(np.floor(sat_psize/32))
#d11 = Linear(csize*2*128,512)
#d12 = Linear(512, 256)
#d13 = Linear(256,64)

