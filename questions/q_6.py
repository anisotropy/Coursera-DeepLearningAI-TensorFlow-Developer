# sign-mnist

import tensorflow as tf
from tensorflow.keras import layers, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications.inception_v3 import InceptionV3
import os
import numpy as np
import csv
import matplotlib.pyplot as mpplot


# Prepare Data
def reshape_1d(values_1d, shape):
    return np.array(values_1d, dtype='float32').reshape(shape)


def image_generators(x_train, y_train, x_valid, y_valid, rescale, batch_size):
    datagen_train = ImageDataGenerator(
        rescale=rescale,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rotation_range=40,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    gen_train = datagen_train.flow(x_train, y_train, batch_size=batch_size, shuffle=True)

    datagen_valid = ImageDataGenerator(rescale=rescale)
    gen_valid = datagen_valid.flow(x_valid, y_valid, batch_size=batch_size, shuffle=True)

    return gen_train, gen_valid


x_train, y_train = [], []
x_valid, y_valid = [], []
target_size = (28, 28)
input_shape = (28, 28, 1)

num_classes = 25

with open('tmp/sign-mnist/sign_mnist_train.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        # y_train.append(tf.keras.utils.to_categorical(int(row[0]), num_classes))
        # values = np.array(row[1:], dtype='float32').reshape(target_size)
        # x_train.append(np.expand_dims(values, axis=-1))
        y_train.append(int(row[0]))
        x_train.append(np.array(row[1:], dtype='float32').reshape(input_shape))

with open('tmp/sign-mnist/sign_mnist_test.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        y_valid.append(int(row[0]))
        x_valid.append(np.array(row[1:], dtype='float32').reshape(input_shape))

x_train = np.array(x_train)
y_train = np.array(y_train)
x_valid = np.array(x_valid)
y_valid = np.array(y_valid)

gen_train, gen_valid = image_generators(x_train, y_train, x_valid, y_valid, rescale=1/255, batch_size=32)


# Model
model = tf.keras.models.Sequential([
    Input(shape=input_shape),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    # layers.Dropout(0.2),
    layers.Dense(num_classes, activation='softmax')
])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['acc']
)

histories = []


# Train
def callback_of_stop_training(condition, message='Cancel training...'):
    class StopTraining(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if condition(logs):
                print('\n{}'.format(message))
                self.model.stop_training = True

    return StopTraining()


stop_training = callback_of_stop_training(
    lambda logs: logs.get('acc') > 0.9 and logs.get('val_acc') > 0.9
)
history = model.fit(gen_train, validation_data=gen_valid, epochs=50, callbacks=[stop_training])
histories.append(history.history)


# History
def flat_histories(histories):
    history = {}
    for h in histories:
        for metric, values in h.items():
            if not history.get(metric):
                history[metric] = []
            for value in values:
                history[metric].append(value)
    return history


def plot_history(history, metrics=('loss',)):
    mpplot.figure(figsize=(10, 6))
    epochs = range(len(history[metrics[0]]))
    for metric in metrics:
        mpplot.plot(epochs, history[metric], label=metric)
    mpplot.legend()


my_history = flat_histories(histories)
plot_history(my_history, ('acc', 'val_acc'))




