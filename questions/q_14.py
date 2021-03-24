# https://storage.googleapis.com/laurencemoroney-blog.appspot.com/irish-lyrics-eof.txt


import tensorflow as tf
from tensorflow.keras import layers, Input, regularizers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as mpplot


# Get Data
texts = []
with open('/tmp/irish-lyrics-eof.txt') as file:
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


x_train, y_train, num_words, sequence_length, tokenizer = texts_to_n_gram_sequences(texts)


# Model
embedding_dim = 100
model = tf.keras.models.Sequential([
    Input(shape=(sequence_length,)),
    layers.Embedding(num_words, embedding_dim),
    # layers.Conv1D(32, 5, activation='relu'),
    # layers.MaxPooling1D(4),
    # layers.Bidirectional(layers.LSTM(32, return_sequences=True)),
    layers.Bidirectional(layers.LSTM(150)),
    # layers.Dense(32, activation='relu'),
    layers.Dense(num_words, activation='softmax')
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=0.01),
    loss='categorical_crossentropy',
    metrics=['acc']
)
histories = []


# Train
def callback_of_stop_training(condition, message='Cancel training...'):
    class StopTraining(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if condition(logs):
                print('\n{}'.format(message))
                self.model.stop_training = True

    return StopTraining()


stop_training = callback_of_stop_training(
    lambda logs: logs.get('acc') > 0.8
)
history = model.fit(x_train, y_train, epochs=10, callbacks=[stop_training])
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
plot_history(my_history, ('acc',))



