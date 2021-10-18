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

basic_cnn.add(Conv2D(64, 3, padding="same", activation="relu"))
basic_cnn.add(MaxPool2D())
basic_cnn.add(Dropout(0.4))

basic_cnn.add(Flatten())
basic_cnn.add(Dense(128,activation="relu"))
basic_cnn.add(Dense(1, activation="sigmoid"))


#1st Convolutional Layer
AlexNet = Sequential()
AlexNet.add(Conv2D(filters=96, input_shape=(224,224,3), kernel_size=(11,11), strides=(4,4), padding='same'))
AlexNet.add(BatchNormalization())
AlexNet.add(Activation('relu'))
AlexNet.add(MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same'))

#2nd Convolutional Layer
AlexNet.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1,1), padding='same'))
AlexNet.add(BatchNormalization())
AlexNet.add(Activation('relu'))
AlexNet.add(MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same'))

#3rd Convolutional Layer
AlexNet.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
AlexNet.add(BatchNormalization())
AlexNet.add(Activation('relu'))

#4th Convolutional Layer
AlexNet.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
AlexNet.add(BatchNormalization())
AlexNet.add(Activation('relu'))

#5th Convolutional Layer
AlexNet.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))
AlexNet.add(BatchNormalization())
AlexNet.add(Activation('relu'))
AlexNet.add(MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same'))

#Passing it to a Fully Connected layer
AlexNet.add(Flatten())
# 1st Fully Connected Layer
AlexNet.add(Dense(4096, activation='relu'))
AlexNet.add(BatchNormalization())
AlexNet.add(Dropout(0.4))

#2nd Fully Connected Layer
AlexNet.add(Dense(4096))
AlexNet.add(BatchNormalization())
AlexNet.add(Activation('relu'))
AlexNet.add(Dropout(0.4))

#3rd Fully Connected Layer
AlexNet.add(Dense(1000))
AlexNet.add(BatchNormalization())
AlexNet.add(Activation('relu'))
#Add Dropout
AlexNet.add(Dropout(0.4))

#Output Layer
AlexNet.add(Dense(1, activation="sigmoid")) # would change in the multi-class case

