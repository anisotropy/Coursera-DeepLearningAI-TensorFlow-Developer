# cats-v-dogs


import tensorflow as tf
from tensorflow.keras import layers, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications.inception_v3 import InceptionV3
import os
import numpy as np
import matplotlib.pyplot as mpplot


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


dir_train = '/tmp/cats-v-dogs/training'
dir_valid = '/tmp/cats-v-dogs/testing'
target_size = (150, 150)
input_shape = (*target_size, 3)

gen_train, gen_valid = image_generators_dir(
    dir_train, dir_valid, target_size,
    class_mode='binary',
    rescale=1/255,
    batch_size=32
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


inception_weights_path = 'inception_weights.h5'
inception_weights_url = 'https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
tf.keras.utils.get_file(os.path.abspath(inception_weights_path), origin=inception_weights_url)
last_layer_name = 'mixed7'

model = model_with_inception_v3(
    inception_weights_path, input_shape, last_layer_name, [
        layers.Flatten(),
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ]
)

# model = tf.keras.models.Sequential([
#     Input(shape=input_shape),
#     layers.Conv2D(32, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(64, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(128, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(128, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Flatten(),
#     layers.Dense(512, activation='relu'),
#     layers.Dropout(0.5),
#     layers.Dense(1, activation='sigmoid')
# ])

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
history = model.fit(gen_train, validation_data=gen_valid, epochs=10, callbacks=[stop_training])
histories.append(history.history)