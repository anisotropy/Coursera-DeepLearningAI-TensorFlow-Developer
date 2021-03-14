import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


mnist = tf.keras.datasets.mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

training_images = np.expand_dims(training_images, axis=-1)
# ALTER: training_images = training_images.reshape(60000, 28, 28, 1)
training_images = training_images / 255

test_images = np.expand_dims(test_images, axis=-1)
# ALTER: test_images = test_images.reshape(10000, 28, 28, 1)
test_images = test_images / 255

num_class = len(np.unique(training_labels))


model = tf.keras.models.Sequential([
    layers.Conv2D(filters=64, kernel_size=(3, 3), input_shape=(28, 28, 1)),
    layers.MaxPool2D(pool_size=(2, 2)),
    layers.Conv2D(filters=64, kernel_size=(3, 3)),
    layers.MaxPool2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(units=128, activation='relu'),
    layers.Dense(units=num_class, activation='softmax')
])
model.summary()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


class MyCallbacks(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') >= 0.98:
            print('Reaches 98% accuracy so cancelling training!!')
            self.model.stop_training = True


callback = MyCallbacks()
history = model.fit(training_images, training_labels, epochs=10, callbacks=[callback])
