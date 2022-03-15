import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras import layers
import numpy as np

ts_datagen = ImageDataGenerator(
    rescale= 1./255,
    shear_range= 0.2,
    zoom_range= 0.2,
    horizontal_flip= True)

training_set = ts_datagen.flow_from_directory(
    'dataset_2/training_set',
    target_size= (64,64),
    batch_size= 32,
    class_mode= 'binary')

vs_datagen = ImageDataGenerator(rescale=1./255)

validation_set = vs_datagen.flow_from_directory(
    'dataset_2/test_set',
    target_size= (64,64),
    batch_size= 32,
    class_mode= 'binary')

cnn = Sequential()
cnn.add(layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
cnn.add(layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(layers.Flatten())
cnn.add(layers.Dense(units=128, activation='relu'))
cnn.add(layers.Dense(units=1, activation='sigmoid'))

cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn.fit(x=training_set, validation_data=validation_set, epochs= 25)
cnn.save('model/cnn_model.h5', overwrite=True)

