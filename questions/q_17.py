# sunspots.csv

import tensorflow as tf
from tensorflow.keras import layers, Input
import csv
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


# Get Data
series = []

with open('tmp/sunspots.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        series.append(float(row[2]))


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


split_size = int(len(series) * 0.8)
window_size = 60
shuffle_buffer_size = 1000
batch_size = 32

ds_train = windowed_dataset(series[:split_size], window_size, shuffle_buffer_size, batch_size)
ds_valid = windowed_dataset(series[split_size-window_size:], window_size, shuffle_buffer_size, batch_size)


# Model
learning_rate = 1e-3
model = tf.keras.models.Sequential([
    Input(shape=(None, 1)),
    layers.Conv1D(32, 5, padding='causal', activation='relu'),
    layers.Bidirectional(layers.LSTM(32, return_sequences=True)),
    layers.Bidirectional(layers.LSTM(32)),
    layers.Dense(32, activation='relu'),
    layers.Dense(10, activation='relu'),
    layers.Dense(1)
])
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(lr=learning_rate),
    loss=tf.keras.losses.Huber(),
    metrics=['mae']
)
histories = []


# Train
epochs = 70
# lr_schedule = tf.keras.callbacks.LearningRateScheduler(
#     lambda epoch: learning_rate * 10 ** (epoch / 10)
# )
# history = model.fit(ds_train, validation_data=ds_valid, epochs=epochs, callbacks=[lr_schedule])
# mpplot.semilogx(history.history['lr'], history.history['loss'])
history = model.fit(ds_train, validation_data=ds_valid, epochs=epochs)
histories.append(history.history)
plot_history(history.history, ('mae', 'val_mae'))


# History
saved_history = flat_histories(histories)
plot_history(saved_history, ('mae', 'val_mae'))

plot([saved_history['mae'][:epochs], history.history['mae']], ['old', 'new'])
plot([saved_history['val_mae'][:epochs], history.history['val_mae']], ['old', 'new'])


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
plot([series[split_size:], forecast], ['true', 'forecast'])


# Save Model
model.save('model_q17_1.h5')