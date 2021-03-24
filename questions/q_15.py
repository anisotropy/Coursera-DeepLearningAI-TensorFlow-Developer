# sonnets.txt


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
texts = []
with open('tmp/sonnets.txt', 'r') as file:
    for line in file:
        texts.append(line)


# Prepare Data
def texts_to_n_gram_sequences(texts, oov_token=None):
    tokenizer = Tokenizer(oov_token=oov_token)
    tokenizer.fit_on_texts(texts)
    num_words = len(tokenizer.word_index) + 1
    seqs = tokenizer.texts_to_sequences(texts)

    input_seqs = []
    max_length = 0
    for seq in seqs:
        seq_len = len(seq)
        max_length = (lambda x: x if x > max_length else max_length)(seq_len)
        for i in range(1, seq_len):
            input_seqs.append(seq[:i+1])

    padded_seqs = pad_sequences(input_seqs, maxlen=max_length)

    x_train = padded_seqs[:, :-1]
    y_train = tf.keras.utils.to_categorical(padded_seqs[:, -1], num_classes=num_words)

    sequence_length = max_length - 1

    return x_train, y_train, num_words, sequence_length, tokenizer


x_data, y_data, num_words, sequence_length, tokenizer = texts_to_n_gram_sequences(texts)


# Model
clear_session()

embedding_dim = 100
model = tf.keras.models.Sequential([
    Input(shape=(sequence_length,)),
    layers.Embedding(num_words, embedding_dim),
    # layers.Conv1D(64, 5, activation='relu'),
    # layers.MaxPooling1D(4),
    # layers.Bidirectional(layers.LSTM(32, return_sequences=True)),
    layers.Bidirectional(layers.LSTM(32)),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_words, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
histories = []


# Model -- new
clear_session()

embedding_dim = 100
model = tf.keras.models.Sequential([
    Input(shape=(sequence_length,)),
    layers.Embedding(num_words, embedding_dim),
    # layers.Conv1D(64, 5, activation='relu'),
    # layers.MaxPooling1D(4),
    # layers.Bidirectional(layers.LSTM(32, return_sequences=True)),
    layers.Bidirectional(layers.LSTM(32)),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_words, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
histories = []


# Train
epoch_unit = 60
stop_training = callback_of_stop_training(lambda logs: logs.get('acc') > 0.8 and logs.get('val_acc') > 0.8)
history = model.fit(
    x_data, y_data, validation_split=0.2, shuffle=True,
    epochs=epoch_unit, callbacks=[stop_training]
)
histories.append(history.history)
plot_history(history.history, ('acc', 'val_acc'))

# History
saved_history = flat_histories(histories)
plot_history(saved_history, ('acc', 'val_acc'))

metric = 'acc'
plot([saved_history[metric][:epoch_unit], history.history[metric]], ['old', 'new'])


# Save Model
model.save('model_q15_2.h5')
