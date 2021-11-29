import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout, Activation, BatchNormalization
from tensorflow.keras import Model, regularizers
from config import get_args

args = get_args()

fully_connected = Sequential([Flatten(input_shape = (224,224,3)), 
                                    Dense(128, activation=tf.nn.relu), 
                                    Dense(1, activation=tf.nn.sigmoid)])



basic_cnn = Sequential()
basic_cnn.add(Conv2D(32,3,padding="same", activation="relu", input_shape=(224,224,3)))
basic_cnn.add(MaxPool2D())

basic_cnn.add(Conv2D(32, 3, padding="same", activation="relu"))
basic_cnn.add(MaxPool2D())

basic_cnn.add(Dropout(0.4))
basic_cnn.add(Conv2D(64, 3, padding="same", activation="relu"))
basic_cnn.add(MaxPool2D())

basic_cnn.add(Flatten())
basic_cnn.add(Dense(128,activation="relu"))
basic_cnn.add(Dense(1, activation="sigmoid"))

#1st Convolutional Layer
# (3) Create a sequential model
AlexNet = Sequential()

# 1st Convolutional Layer
AlexNet.add(Conv2D(filters=24, input_shape=(224,224,3), kernel_size=3, padding='same', activation="relu"))
AlexNet.add(MaxPool2D(pool_size=2))
# AlexNet.add(BatchNormalization())


# 2nd Convolutional Layer
AlexNet.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
AlexNet.add(MaxPool2D(pool_size=2))
# AlexNet.add(BatchNormalization())


# 3rd Convolutional Layer
AlexNet.add(Conv2D(filters=96, kernel_size=(3,3), padding='same', activation='relu'))
# AlexNet.add(BatchNormalization())


# 4th Convolutional Layer
AlexNet.add(Conv2D(filters=96, kernel_size=(3,3), padding='same', activation='relu'))
# AlexNet.add(BatchNormalization())


# 5th Convolutional Layer
AlexNet.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu'))
AlexNet.add(MaxPool2D(pool_size=2))
# AlexNet.add(BatchNormalization())


# Passing it to a dense layer
AlexNet.add(Flatten())


# 1st Dense Layer
AlexNet.add(Dense(800, activation='relu'))
AlexNet.add(Dropout(0.4))
# AlexNet.add(BatchNormalization())


# 2nd Dense Layer
AlexNet.add(Dense(800, activation='relu'))
AlexNet.add(Dropout(0.4))
# AlexNet.add(BatchNormalization())

#  output Layer
AlexNet.add(Dense(1, activation='sigmoid'))

#AlexNet.summary()

##### XCEPTION #####
xception = tf.keras.applications.Xception(
    include_top=False,
    weights="imagenet",
    input_shape=(224,224, 3)
)

for layer in xception.layers:
  layer.trainable = False

##### INCEPTION #####
inceptionv3 = tf.keras.applications.InceptionV3(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

for layer in inceptionv3.layers:
  layer.trainable = False

##### RESNET #####
resnet = tf.keras.applications.ResNet50(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

for layer in resnet.layers:
  layer.trainable = False
  
def output_layer(model):
  # Flatten the output layer to 1 dimension
  x = Flatten()(model.output)
  # Add a first fully connected layer with 800 hidden units and ReLU activation
  x = Dense(800, activation='relu', kernel_regularizer=regularizers.l2(args.reg),
    bias_regularizer=regularizers.l2(args.reg),
    activity_regularizer=regularizers.l2(args.reg))(x)
  # Add a dropout rate of 0.2
  x = Dropout(args.dropout)(x)

  # Add a second fully connected layer with 800 hidden units and ReLU activation
  x = Dense(800, activation='relu', kernel_regularizer=regularizers.l2(args.reg),
    bias_regularizer=regularizers.l2(args.reg),
    activity_regularizer=regularizers.l2(args.reg))(x)
  x = Dropout(args.dropout)(x)

  # Add a final sigmoid layer for classification
  x = Dense(1, activation='sigmoid')(x)
  return x

Inception = Model(inceptionv3.input, output_layer(inceptionv3))
Xception = Model(xception.input, output_layer(xception))
ResNet = Model(resnet.input, output_layer(resnet))

