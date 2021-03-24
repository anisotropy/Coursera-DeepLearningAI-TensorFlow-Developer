# horse-or-human/horses

import tensorflow as tf
from tensorflow.keras import layers, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications.inception_v3 import InceptionV3
import os
import matplotlib.pyplot as mpplot
import numpy as np


# Check Image Size
def load_image(directory=None, nth=0, path=None, target_size=None):
    if path is None:
        image_path = os.path.join(directory, os.listdir(directory)[0])
    else:
        image_path = path
    return load_img(image_path, target_size=target_size)


print(load_image('/tmp/horse-or-human/horses', 0))


# Prepare Data
def image_generators_dir(dir_train, dir_valid, target_size, class_mode, rescale, batch_size):
    datagen_train = ImageDataGenerator(
        rescale=rescale,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        rotation_range=40
    )
    gen_train = datagen_train.flow_from_directory(
        dir_train,
        target_size=target_size, class_mode=class_mode, batch_size=batch_size,
        shuffle=True
    )

    datagen_valid = ImageDataGenerator(rescale=rescale)
    gen_valid = datagen_valid.flow_from_directory(
        dir_valid,
        target_size=target_size, class_mode=class_mode, batch_size=batch_size,
        shuffle=True
    )

    return gen_train, gen_valid


dir_train = '/tmp/horse-or-human'
dir_valid = '/tmp/validation-horse-or-human'

target_size = (150, 150)
input_shape = (*target_size, 3)
gen_train, gen_valid = image_generators_dir(
    dir_train, dir_valid, target_size, class_mode='binary', rescale=1/255, batch_size=32
)


# Model
def model_with_inception_v3(weights_path, input_shape, last_layer_name, successive_layers):
    pretrained_model = InceptionV3(input_shape=input_shape, include_top=False, weights=None)
    pretrained_model.load_weights(weights_path)
    for layer in pretrained_model.layers:
        layer.trainable = False
    output = pretrained_model.get_layer(last_layer_name).output

    for layer in successive_layers:
        output = layer(output)

    return tf.keras.Model(pretrained_model.input, output)


def fetch_from_url(url, filepath):
    tf.keras.utils.get_file(os.path.abspath(filepath), origin=url)

fetch_from_url(
    'https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
    'conv_weights.h5'
)

model = model_with_inception_v3('conv_weights.h5', input_shape, 'mixed7', [
    layers.Flatten(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
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

history = model.fit(
    gen_train,
    validation_data=gen_valid,
    epochs=10,
    callbacks=[stop_training]
)

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