# sign-mnist.csv

import tensorflow as tf
from tensorflow.keras import layers, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications.inception_v3 import InceptionV3
import os
import numpy as np
import matplotlib.pyplot as mpplot
import csv


def clear_session():
    tf.keras.backend.clear_session()
    tf.random.set_seed(51)
    np.random.seed(51)


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


def plot(ys, labels=None):
    mpplot.figure(figsize=(10, 6))
    x = range(len(ys[0]))
    for i in range(len(ys)):
        if labels is not None and len(labels) == len(ys):
            label = labels[i]
        else:
            label = None
        mpplot.plot(x, ys[i], label=label)
    mpplot.legend()


def callback_of_stop_training(condition, message='Cancel training...'):
    class StopTraining(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if condition(logs):
                print('\n{}'.format(message))
                self.model.stop_training = True

    return StopTraining()


# Get Data
def reshape_1d(values_1d, shape):
    return np.array(values_1d, dtype='float32').reshape(shape)


image_shape = (28, 28, 1)
labels_train, image_train = [], []
labels_test, image_test = [], []

with open('tmp/sign-mnist/sign_mnist_train.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        labels_train.append(int(row[0]))
        image_train.append(reshape_1d(row[1:], image_shape))

with open('tmp/sign-mnist/sign_mnist_test.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        labels_test.append(int(row[0]))
        image_test.append(reshape_1d(row[1:], image_shape))


x_train = np.array(image_train)
y_train = np.array(labels_train)
x_valid = np.array(image_test)
y_valid = np.array(labels_test)


# Prepare Data
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


gen_train, gen_valid = image_generators(
    x_train, y_train, x_valid, y_valid,
    rescale=1/255, batch_size=32
)


# Model -- new
model = tf.keras.models.Sequential([
    Input(shape=image_shape),
    layers.Conv2D(32, (3, 3), activation='relu', ),
    layers.MaxPooling2D(pool_size=(2, 2)),
    # layers.Conv2D(32, (3, 3), activation='relu'),
    # layers.MaxPooling2D(pool_size=(2, 2)),
    # layers.Conv2D(64, (3, 3), activation='relu'),
    # layers.MaxPooling2D(pool_size=(2, 2)),
    # layers.Conv2D(64, (3, 3), activation='relu'),
    # layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(1024, activation='relu'),
    # layers.Dropout(0.2),
    layers.Dense(25, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
histories = []


# Train
stop_training = callback_of_stop_training(lambda logs: logs.get('acc') > 0.8 and logs.get('val_acc') > 0.8)
history = model.fit(gen_train, validation_data=gen_valid, epochs=10, callbacks=[stop_training])
histories.append(history.history)
plot_history(history.history, ('acc', 'val_acc'))


# History
saved_history = flat_histories(histories)
plot_history(saved_history, ('acc', 'val_acc'))

plot([saved_history['val_acc'][:10], history.history['val_acc']], ['old', 'new'])


# Save Model
model.save('model_q6_1.h5')
