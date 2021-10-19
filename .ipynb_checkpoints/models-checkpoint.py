import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout 


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


