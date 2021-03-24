# imdb, info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)


import tensorflow as tf
from tensorflow.keras import layers, Input, regularizers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow_datasets as tfds
import numpy as np
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


# Prepare Data
def load_imdb_subwords():
    imdb, info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)
    tokenizer = info.features['text'].encoder
    return imdb, tokenizer


def padded_batch_ds(raw_ds, batch_size, shuffle_buffer_size=None):
    if shuffle_buffer_size is None:
        shuffled_ds = raw_ds
    else:
        shuffled_ds = raw_ds.shuffle(shuffle_buffer_size)
    ds = shuffled_ds.padded_batch(batch_size, tf.compat.v1.data.get_output_shapes(raw_ds))
    return ds


imdb, tokenizer = load_imdb_subwords()
ds_train = padded_batch_ds(imdb['train'], batch_size=32, shuffle_buffer_size=1000)
ds_valid = padded_batch_ds(imdb['test'], batch_size=32)


# Model
embedding_dim = 32
model = tf.keras.models.Sequential([
    layers.Embedding(tokenizer.vocab_size, embedding_dim),
    layers.Conv1D(16, 5, activation='relu'),
    layers.MaxPooling1D(4),
    # layers.Bidirectional(layers.LSTM(16, return_sequences=True)),
    layers.Bidirectional(layers.LSTM(16)),
    layers.Dense(16, activation='relu'),
    # layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
histories = []


# Train
epoch_unit = 10
stop_training = callback_of_stop_training(lambda logs: logs.get('acc') > 0.8 and logs.get('val_acc') > 0.8)
history = model.fit(ds_train, validation_data=ds_valid, epochs=epoch_unit, callbacks=[stop_training])
histories.append(history.history)
plot_history(history.history, ('acc', 'val_acc'))


# History
saved_history = flat_histories(histories)
plot_history(saved_history, ('acc', 'val_acc'))


plot([saved_history['acc'][:epoch_unit], history.history['acc']], ['old', 'new'])
plot([saved_history['val_acc'][:epoch_unit], history.history['val_acc']], ['old', 'new'])


# Save Model
model.save('model_q10_1.h5')
