import tensorflow as tf
from tensorflow.keras import layers, Input


def windowed_dataset(train_series, window_size, shuffle_buffer_size, batch_size):
    series = tf.expand_dims(train_series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer_size)
    ds = ds.map(lambda w: (w[:-1], w[-1]))
    ds = ds.batch(batch_size).prefetch(1)
    return ds


def layers_1():
    return [
        layers.Conv1D(32, 5, padding='causal', activation='relu'),
        layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
        layers.Bidirectional(layers.LSTM(64)),
        layers.Dense(30, activation='relu'),
        layers.Dense(10, activation='relu'),
    ]


def layers_2():
    return [
        layers.Bidirectional(layers.LSTM(32, return_sequences=True)),
        layers.Bidirectional(layers.LSTM(32)),
    ]


def layers_3():
    return [
        layers.SimpleRNN(40, return_sequences=True),
        layers.SimpleRNN(40),
    ]


def predict(model, valid_series, window_size, batch_size):
    series = tf.expand_dims(valid_series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(batch_size).prefetch(1)
    results = model.predict(ds)
    return results.flatten()[:-1]
