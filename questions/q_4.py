# !wget --no-check-certificate \
#     https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip \
#     -O /tmp/cats_and_dogs_filtered.zip


import tensorflow as tf
from tensorflow.keras import layers, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications.inception_v3 import InceptionV3
import os
import numpy as np
import matplotlib.pyplot as mpplot


# Get Data
dir_train = '/tmp/cats_and_dogs_filtered/train'
dir_valid = '/tmp/cats_and_dogs_filtered/validation'

def load_image(directory=None, nth=0, path=None, target_size=None):
    if path is None:
        image_path = os.path.join(directory, os.listdir(directory)[0])
    else:
        image_path = path
    return load_img(image_path, target_size=target_size)


print(load_image(dir_train + '/cats', 0))


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


target_size = (300, 300)

gen_train, gen_valid = image_generators_dir(
    dir_train, dir_valid, target_size, class_mode='binary', rescale=1/255, batch_size=32
)


# Model
model = tf.keras.models.Sequential([
    Input(shape=(300, 300, 3)),
    layers.Conv2D(16, (3, 3), activation='relu',),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.2),
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
    lambda logs: logs.get('acc') > 0.8 and logs.get('val_acc') > 0.8
)

history = model.fit(
    gen_train, validation_data=gen_valid, epochs=40, callbacks=[stop_training]
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


# Save Model
model.save('model_q4_1.h5')