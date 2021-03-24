# sunspots.csv
# https://storage.googleapis.com/laurencemoroney-blog.appspot.com/Sunspots.csv


import csv
import tensorflow as tf
from tensorflow.keras import layers, Input
import numpy as np
import matplotlib.pyplot as mpplot


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


# Get Data
series = []
with open('tmp/sunspots.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        series.append(float(row[2]))

split_size = int(len(series) * 0.8)


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


window_size = 60
shuffle_buffer_size = 1000
batch_size = 32

ds_train = windowed_dataset(series[:split_size], window_size, shuffle_buffer_size, batch_size)


# Model
tf.keras.backend.clear_session()
tf.random.set_seed(51)
np.random.seed(51)

learning_rate = 1e-3
model = tf.keras.models.Sequential([
    Input(shape=(None, 1)),
    # layers.Conv1D(32, 5, padding='causal', activation='relu'),
    layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
    layers.Bidirectional(layers.LSTM(64)),
    layers.Dense(30, activation='relu'),
    # layers.Dense(10, activation='relu'),
    layers.Dense(1)
])
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(lr=learning_rate),
    loss=tf.keras.losses.Huber(),
    metrics=['mae']
)

histories = []


# Train

# lr_schedule = tf.keras.callbacks.LearningRateScheduler(
#     lambda epoch: learning_rate * 10 ** (epoch / 20)
# )
# history = model.fit(ds_train, epochs=100, callbacks=[lr_schedule])
# mpplot.semilogx(history.history['lr'], history.history['loss'])

history = model.fit(ds_train, epochs=10)
histories.append(history.history)


# History
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


forecast = predict(model, series[split_size-window_size:], window_size, batch_size)
mpplot.figure(figsize=(10, 6))
time = range(len(forecast))
mpplot.plot(time, series[split_size:], label='true')
mpplot.plot(time, forecast, label='forecast')
mpplot.legend()

print('MAE:', tf.keras.metrics.mean_absolute_error(series[split_size:], forecast).numpy())