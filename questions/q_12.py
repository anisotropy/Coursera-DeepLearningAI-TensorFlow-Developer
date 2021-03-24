# 12. 3-5
# training_cleaned.csv
# embedding weights: glove.6B.100d.txt

import tensorflow as tf
from tensorflow.keras import layers, Input, regularizers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as mpplot
import csv


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


def callback_of_stop_training(condition, message='Cancel training...'):
    class StopTraining(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if condition(logs):
                print('\n{}'.format(message))
                self.model.stop_training = True

    return StopTraining()


# Get Data
texts, labels = [], []

with open('tmp/training_cleaned.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        texts.append(row[5])
        labels.append(row[0])

labels = [0 if label == '0' else 1 for label in labels]


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


max_length = 15
padded_seqs, num_words, tokenizer = texts_to_sequences(texts, max_length)

x_data = padded_seqs
y_data = np.array(labels)


# Model
embedding_dim = 32
model = tf.keras.models.Sequential([
    Input(shape=(max_length,)),
    layers.Embedding(num_words, embedding_dim),
    layers.GlobalAveragePooling1D(),
    # layers.Conv1D(64, 5, activation='relu'),
    # layers.MaxPooling1D(4),
    # layers.Bidirectional(layers.LSTM(32, return_sequences=True)),
    # layers.Bidirectional(layers.LSTM(32)),
    # layers.Dense(num_words / 2, activation='relu'),
    # layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
histories = []


# Train
stop_training = callback_of_stop_training(lambda logs: logs.get('acc') > 0.8 and logs.get('val_acc'))
history = model.fit(
    x_data, y_data, shuffle=True, validation_split=0.2, epochs=10, callbacks=[stop_training]
)
histories.append(history.history)


# History
saved_history = flat_histories(histories)
plot_history(saved_history, ('acc', 'val_acc'))




