# horse-or-human


import tensorflow as tf
from tensorflow.keras import layers, Input, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications.inception_v3 import InceptionV3
import os
import numpy as np
import matplotlib.pyplot as mpplot


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


# Gat Data
def load_image(directory=None, nth=0, path=None, target_size=None):
    if path is None:
        image_path = os.path.join(directory, os.listdir(directory)[0])
    else:
        image_path = path
    return load_img(image_path, target_size=target_size)


print(load_image('tmp/horse-or-human/train/humans', 0))


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


dir_train, dir_valid, target_size = 'tmp/horse-or-human/train', 'tmp/horse-or-human/validation', (150, 150)
input_shape = (*target_size, 3)

gen_train, gen_valid = image_generators_dir(
    dir_train, dir_valid, target_size,
    class_mode='binary', rescale=1/255, batch_size=32
)


# Model
def model_with_inception_v3(weights_path, input_shape, last_layer_name, successive_layers):
    # wights_url = 'https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
    # last_layer_name = 'mixed7'
    pretrained_model = InceptionV3(input_shape=input_shape, include_top=False, weights=None)
    pretrained_model.load_weights(weights_path)
    for layer in pretrained_model.layers:
        layer.trainable = False
    output = pretrained_model.get_layer(last_layer_name).output

    for layer in successive_layers:
        output = layer(output)

    return tf.keras.Model(pretrained_model.input, output)


inception_weights_url = 'https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
inception_weights_path = 'tmp/inception_weights.h5'
tf.keras.utils.get_file(os.path.abspath(inception_weights_path), origin=inception_weights_url)

model = model_with_inception_v3(inception_weights_path, input_shape, 'mixed7', [
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['acc']
)
histories = []


# Train
epoch_unit = 10
stop_training = callback_of_stop_training(lambda logs: logs.get('acc') > 0.8 and logs.get('val_acc') > 0.8)
history = model.fit(gen_train, validation_data=gen_valid, epochs=epoch_unit, callbacks=[stop_training])
histories.append(history.history)
plot_history(history.history, ('acc', 'val_acc'))


# History
saved_history = flat_histories(histories)
plot_history(saved_history, ('acc', 'val_acc'))


plot([saved_history['acc'][:epoch_unit], history.history['acc']], ['old', 'new'])
plot([saved_history['val_acc'][:epoch_unit], history.history['val_acc']], ['old', 'new'])


# Save Model
model.save('model_q3_1.h5')
