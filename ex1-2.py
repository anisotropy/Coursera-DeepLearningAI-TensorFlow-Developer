import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt


mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()


plt.imshow(training_images[0])


training_images = training_images / 255
test_images = test_images / 255

num_classes = len(np.unique(training_labels))
print('num_classes:', num_classes)


model = tf.keras.models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    # ALTER: layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        # ALTER: if logs.get('loss') < 0.4:
        if logs.get('accuracy') >= 0.9:
            # ALTER: print('\nReaches 40% loss so cancelling training!!')
            print('\nReaches 90% accuracy so cancelling training!!')
            self.model.stop_training = True


callback = MyCallback()
model.fit(training_images, training_labels, epochs=10, callbacks=[callback])


model.evaluate(test_images, test_labels)

print(np.argmax(model.predict(test_images[0][np.newaxis])[0]), test_labels[0])
