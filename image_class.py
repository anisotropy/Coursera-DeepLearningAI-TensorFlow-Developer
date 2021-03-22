import tensorflow as tf
from tensorflow.keras import layers, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications.inception_v3 import InceptionV3
import os
import numpy as np


def load_image(directory=None, nth=0, path=None, target_size=None):
    if path is None:
        image_path = os.path.join(directory, os.listdir(directory)[0])
    else:
        image_path = path
    return load_img(image_path, target_size=target_size)


def reshape_1d(values_1d, shape):
    return np.array(values_1d, dtype='float32').reshape(shape)


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


def model_with_inception_v3(weights_path, input_shape, last_layer_name, successive_layers):
    pretrained_model = InceptionV3(input_shape=input_shape, include_top=False, weights=None)
    pretrained_model.load_weights(weights_path)
    for layer in pretrained_model.layers:
        layer.trainable = False
    output = pretrained_model.get_layer(last_layer_name).output

    for layer in successive_layers:
        output = layer(output)

    return tf.keras.Model(pretrained_model.input, output)


def layers_1():
    return [
        layers.Conv2D(16, (3, 3), activation='relu',),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.2),
    ]


def layers_2():
    return [
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.5),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
    ]


def predict_class(model, image):
    rescale = 1/255
    test_data = img_to_array(image)
    test_data = np.expand_dims(test_data, axis=0)
    test_data = test_data * rescale
    result = model.predict(test_data)
    predicted_class = np.argmax(result[0])
    return predicted_class
