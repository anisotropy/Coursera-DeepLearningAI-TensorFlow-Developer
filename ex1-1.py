import tensorflow as tf
import numpy as np


xs = np.array([0., 1., 2., 3., 4., 5., 6.], dtype=float)
ys = np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5])


model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

model.compile(optimizer='sgd', loss='mse')

model.fit(xs, ys, epochs=500)

print(model.predict([10]))
