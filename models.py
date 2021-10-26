import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout, Activation, BatchNormalization


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

AlexNet.summary()


Xception = tf.keras.applications.Xception(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=2,
    classifier_activation="softmax",
)

