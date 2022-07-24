# -*- coding: utf-8 -*-
"""Sunda Aksara Classification.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_esxb_bMzxz30WrO0jG9rEam3mGBDMXf

# Sunda Aksara Classification
"""

# Import raw data into colab
!git clone https://github.com/ridhomujizat/AksaraSundaCNN/

# Import package that we need
import os
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img

# Define root directory 
root_path = '/content/AksaraSundaCNN'

root_path_train = os.path.join(root_path, 'train')
root_path_test = os.path.join(root_path, 'test')

# Do the image augmentation and rescale image
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   width_shift_range = 0.2, height_shift_range = 0.2,
                                   shear_range = 0.2, zoom_range = 0.2, fill_mode='nearest')

train_data = train_datagen.flow_from_directory(
    directory=root_path_train,
    color_mode="grayscale",
    batch_size=30)

validation_datagen = ImageDataGenerator(rescale = 1./255)

validation_data = validation_datagen.flow_from_directory(
    directory=root_path_test,
    target_size=(112,112),
    color_mode="grayscale",
    batch_size=30)

# Define the model
from tensorflow.keras.optimizers import RMSprop
from tensorflow import keras
model = tf.keras.models.Sequential([ 
      tf.keras.layers.Conv2D(256, (5,5), activation = 'relu', padding = 'same', input_shape=(112,112,1)),
      tf.keras.layers.LeakyReLU(alpha=0.2),
      tf.keras.layers.MaxPooling2D(2,2),
      tf.keras.layers.Conv2D(256, (5,5)),
      tf.keras.layers.LeakyReLU(alpha=0.3),
      tf.keras.layers.MaxPooling2D(2,2),
      tf.keras.layers.Conv2D(128, (5,5)),
      tf.keras.layers.LeakyReLU(alpha=0.3),
      tf.keras.layers.MaxPooling2D(2,2),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.GlobalMaxPool2D(),
      tf.keras.layers.Dense(1024),
      tf.keras.layers.LeakyReLU(alpha=0.2),
      tf.keras.layers.Dense(18, activation = 'softmax')
  ])


model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005),
                loss = tf.keras.losses.CategoricalCrossentropy(),
                metrics = ['accuracy'])

# Show model summary
model.summary()

# Train model with 100 epochs
history = model.fit_generator(generator = train_data, validation_data = validation_data, epochs=100)

# Plot the chart for accuracy and loss on both training and validation
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

# Save model into .h5 format
model.save('Aksara_Sunda_Classification.h5',save_format=".h5")

# Convert into TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open('Aksara_Sunda.tflite', 'wb').write(tflite_model)