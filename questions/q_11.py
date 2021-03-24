# bbc-text.csv

import csv
import tensorflow as tf
from tensorflow.keras import layers, Input, regularizers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as mpplot


# Get Data
def remove_stopwords(text, stopwords):
    words_without_stopwords = []
    for word in text.split():
        if not (word in stopwords):
            words_without_stopwords.append(word)
    return ' '.join(words_without_stopwords)


texts, labels = [], []
with open('/tmp/bbc-text.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        labels.append(row[0])
        texts.append(remove_stopwords(row[1].lower().strip(), stopwords))


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


max_length = 120
x_data, num_words, tokenizer = texts_to_sequences(texts, max_length)
y_data, num_classes, _ = texts_to_sequences(labels, 1)


# Model
embedding_dim = 100
model = tf.keras.models.Sequential([
    Input(shape=(None,)),
    layers.Embedding(num_words, embedding_dim),
    # layers.Conv1D(64, 5, activation='relu'),
    # layers.MaxPooling1D(4),
    # layers.Bidirectional(layers.LSTM(32, return_sequences=True)),
    layers.Bidirectional(layers.LSTM(32)),
    layers.Dense(32, activation='relu'),
    # layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])
model.compile(
    optimizer='rmsprop',
    loss='sparse_categorical_crossentropy',
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
    lambda logs: logs.get('acc') > 0.8 and logs.get('val_acc') > 0.8
)
history = model.fit(x_data, y_data, shuffle=True, validation_split=0.2, epochs=50, callbacks=[stop_training])
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
plot_history(my_history, ('acc', 'val_acc'))


# Save Model
model.save('model_q11_1.h5')
