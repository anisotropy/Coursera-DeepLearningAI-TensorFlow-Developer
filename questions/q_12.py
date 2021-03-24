# training_cleaned.csv


import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Input, regularizers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow_datasets as tfds
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
x_data, y_data = [], []
with open('tmp/training_cleaned.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        x_data.append(row[5])
        y_data.append(int(row[0]))

y_data = [0 if d == 0 else 1 for d in y_data]


# Prepare Data
def texts_to_sequences(texts, max_length, tokenizer=None, oov_token=None):
    padding = 'post'
    truncating = 'post'
    if tokenizer is None:
        tokenizer = Tokenizer(oov_token=oov_token)
        tokenizer.fit_on_texts(texts)
    num_words = len(tokenizer.word_index) + 1

    seqs = tokenizer.texts_to_sequences(texts)
    padded_seqs = pad_sequences(seqs, maxlen=max_length, padding=padding, truncating=truncating)

    return padded_seqs, num_words, tokenizer


split_size = int(len(x_data) * 0.8)
max_length = 20
x_train, num_words, tokenizer = texts_to_sequences(x_data[:split_size], max_length)
x_valid, _, _ = texts_to_sequences(x_data[split_size:], max_length, tokenizer)
y_train = np.array(y_data[:split_size])
y_valid = np.array(y_data[split_size:])


# Model
clear_session()

embedding_dim = 32
model = tf.keras.models.Sequential([
    Input(shape=(max_length,)),
    layers.Embedding(num_words, embedding_dim),
    # layers.Conv1D(64, 5, activation='relu'),
    # layers.MaxPooling1D(4),
    # layers.Bidirectional(layers.LSTM(32, return_sequences=True)),
    layers.Bidirectional(layers.LSTM(32)),
    # layers.Dense(num_words / 2, activation='relu'),
    # layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
histories = []


# Train
stop_training = callback_of_stop_training(lambda logs: logs.get('acc') > 0.8 and logs.get('val_acc') > 0.8)
history = model.fit(
    x_train, y_train, validation_data=(x_valid, y_valid), shuffle=True,
    epochs=10, callbacks=[stop_training]
)
histories.append(history.history)


# History
saved_history = flat_histories(histories)
plot_history(saved_history, ('acc', 'val_acc'))

plot([saved_history['val_acc'][:10], history.history['val_acc']], ['old', 'new'])





