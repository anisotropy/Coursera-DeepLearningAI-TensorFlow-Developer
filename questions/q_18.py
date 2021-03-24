# daily-min-temperatures.csv
# https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv


import tensorflow as tf
from tensorflow.keras import layers, Input
import numpy as np
import matplotlib.pyplot as mpplot
import csv
import os


# Get Data
data_url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv'
data_path = 'daily-min-temperatures.csv'
tf.keras.utils.get_file(os.path.abspath(data_path), origin=data_url)

temperatures = []
with open(data_path, 'r') as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        temperatures.append(float(row[1]))


# Prepare Data
def windowed_dataset(train_series, window_size, shuffle_buffer_size, batch_size):
    series = tf.expand_dims(train_series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer_size)
    ds = ds.map(lambda w: (w[:-1], w[-1]))
    ds = ds.batch(batch_size).prefetch(1)
    return ds


tf.keras.backend.clear_session()
tf.random.set_seed(51)
np.random.seed(51)


split_size = int(len(temperatures) * 0.8)
window_size = 30
shuffle_buffer_size = 1000
batch_size = 32

ds_train = windowed_dataset(temperatures[:split_size], window_size, shuffle_buffer_size, batch_size)


# Model
learning_rate = 1e-3
model = tf.keras.models.Sequential([
    Input(shape=(None, 1)),
    layers.Conv1D(64, 5, padding='causal', activation='relu'),
    layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
    layers.Bidirectional(layers.LSTM(128)),
    layers.Dense(30, activation='relu'),
    layers.Dense(10, activation='relu'),
    layers.Dense(1)
])
model.compile(
    optimizer=tf.keras.optimizers.SGD(lr=learning_rate, momentum=0.9),
    loss=tf.keras.losses.Huber(),
    metrics=['mae']
)

histories = []


# Train
# lr_schedule = tf.keras.callbacks.LearningRateScheduler(
#     lambda epoch: learning_rate * 10 ** (epoch / 20)
# )
# history = model.fit(ds_train, epochs=100, callbacks=[lr_schedule])
history = model.fit(ds_train, epochs=150)
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
plot_history(my_history, metrics=('loss', 'mae'))


# Predict
def predict(model, valid_series, window_size, batch_size):
    series = tf.expand_dims(valid_series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(batch_size).prefetch(1)
    results = model.predict(ds)
    return results.flatten()[:-1]


forecast = predict(model, temperatures[split_size-window_size:], window_size, batch_size)

print('MAE: {}'.format(tf.keras.metrics.mean_absolute_error(temperatures[split_size:], forecast).numpy()))

mpplot.figure(figsize=(10, 6))
time = range(len(forecast))
mpplot.plot(time, temperatures[split_size:], label='true')
mpplot.plot(time, forecast, label='forecast')
mpplot.legend()


# Save Model
model.save('model_q18_2.h5')

