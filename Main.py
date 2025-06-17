#import statments
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.mixed_precision import set_global_policy

#building the model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in base_model.layers:
    layer.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(1024, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(train_generator.num_classes, activation='softmax'),
])

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
set_global_policy('float32')

#implementing callbacks
lr_callback = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patience = 2, min_lr = 1e-6, verbose = 1)
early_callback = EarlyStopping(monitor = 'val_loss', patience = 5, restore_best_weights = True)

#training the model
history = model.fit(train_generator, epochs = 10, 
                    validation_data = val_generator, 
                    steps_per_epoch = 50, validation_steps = 25, 
                    callbacks = [lr_callback, early_callback])

#finetuning the model
for layer in base_model.layers[-5:]:
    layer.trainable = True

for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

model.compile(loss = 'categorical_crossentropy', optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-5), metrics = ['accuracy'])
finetuned_history = model.fit(train_generator, epochs = 10, 
                    validation_data = val_generator, 
                    steps_per_epoch = 50, validation_steps = 25, 
                    callbacks = [lr_callback, early_callback])