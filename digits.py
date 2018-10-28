import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Activation, Dropout
from keras.callbacks import TensorBoard
import time
import numpy as np

NAME = 'CNN-64x2+D128-{}'.format(time.time())
tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))
#Load data from MNIST dataset through keras
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Normalize data so all numbers are 0-1
x_train = keras.utils.normalize(x_train, axis=1)
x_test = keras.utils.normalize(x_test, axis=1)

x_train = x_train.reshape(x_train.shape[0], 1, 28, 28).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 1, 28, 28).astype('float32')

model = Sequential()

#First layer, with a convolution of size 16
model.add(Conv2D(64, (3, 3), input_shape=(1, 28, 28), data_format='channels_first'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

#Second layer, with a convolution of size 16
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#Flatten the data and pass it into a Dense layer
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))

model.add(Dense(10))
model.add(Activation('softmax'))

#Create models optimizer and loss
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#Train the model on MNSIT data
model.fit(x_train, y_train, epochs=10, validation_split=0.1, batch_size=128, callbacks=[tensorboard])

#Save the model
model.save('digits_classifier.model')