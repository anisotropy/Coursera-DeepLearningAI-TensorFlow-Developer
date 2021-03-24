# mnist = tf.keras.datasets.fashion_mnist


import tensorflow as tf
from tensorflow.keras import layers, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications.inception_v3 import InceptionV3
import os
import numpy as np
import matplotlib.pyplot as mpplot


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


def callback_of_stop_training(condition, message='Cancel training...'):
    class StopTraining(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if condition(logs):
                print('\n{}'.format(message))
                self.model.stop_training = True

    return StopTraining()


# Get Data
def data_from_mnist(mnist, rescale):
    (images_train, labels_train), (images_valid, labels_valid) = mnist.load_data()

    x_train = np.expand_dims(images_train, axis=-1) * rescale
    y_train = labels_train

    x_valid = np.expand_dims(images_valid, axis=-1) * rescale
    y_valid = labels_valid

    num_class = len(np.unique(x_train))

    image_shape = x_train[0].shape

    return x_train, y_train, x_valid, y_valid, num_class, image_shape


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


x_train, y_train, x_valid, y_valid, num_class, image_shape = \
    data_from_mnist(tf.keras.datasets.fashion_mnist, rescale=1.0)

gen_train, gen_valid = \
    image_generators(x_train, y_train, x_valid, y_valid, rescale=1/255, batch_size=32)

input_shape = (28, 28, 1)


# Model
model = tf.keras.models.Sequential([
    Input(shape=input_shape),
    layers.Conv2D(16, (3, 3), activation='relu', ),
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
    layers.Dense(10, activation='softmax')
])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['acc']
)
histories = []


# Train
stop_training = callback_of_stop_training(lambda logs: logs.get('acc') > 0.8 and logs.get('val_acc') > 0.8)
history = model.fit(gen_train, validation_data=gen_valid, epochs=5, callbacks=[stop_training])
histories.append(history.history)


# History
saved_history = flat_histories(histories)
plot_history(saved_history, ('acc', 'val_acc'))
plot_history(saved_history, ('loss', 'val_loss'))


# Save Model
model.save('model_q2_1.h5')



