# xs = np.array([0., 1., 2., 3., 4., 5., 6.], dtype=float)
# ys = np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5])
# y = x / 2 + 0.5


import tensorflow as tf
from tensorflow.keras import layers, Input
import numpy as np


# Prepare Data
xs = np.array([0., 1., 2., 3., 4., 5., 6.], dtype=float)
ys = np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5], dtype=float)


# Model
model = tf.keras.models.Sequential([
    Input((1,)),
    layers.Dense(1)
])
model.compile(optimizer='sgd', loss='mse')


# Train
model.fit(xs, ys, epochs=100)


# Predict
result = model.predict([7])
print(result)


# Save Model
model.save('model_q1_1.h5')

