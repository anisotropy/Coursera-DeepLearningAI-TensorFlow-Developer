# sarcasm.json

import tensorflow as tf
from tensorflow.keras import layers, Input, regularizers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as mpplot
import json


# Get Data
texts = []
labels = []
with open('/tmp/sarcasm.json', 'r') as file:
    for d in json.load(file):
        texts.append(d['headline'])
        labels.append(d['is_sarcastic'])


# Prepare Data
def texts_to_sequences(texts, max_length, oov_token=None):
    padding = 'post'
    truncating = 'post'
    tokenizer = Tokenizer(oov_token=oov_token)
    tokenizer.fit_on_texts(texts)
    num_words = len(tokenizer.word_index) + 1

    seqs = tokenizer.texts_to_sequences(texts)
    padded_seqs = pad_sequences(seqs, maxlen=max_length, padding=padding, truncating=truncating)

    return padded_seqs, num_words, tokenizer

max_length = 200
padded_seqs, num_words, tokenizer = texts_to_sequences(texts, max_length)

x_data = padded_seqs
y_data = np.array(labels)


# Model
embedding_dim = 32
model = tf.keras.models.Sequential([
    Input(shape=(max_length,)),
    layers.Embedding(num_words, embedding_dim),
    layers.Conv1D(64, 5, activation='relu'),
    layers.MaxPooling1D(4),
    layers.Bidirectional(layers.LSTM(32, return_sequences=True)),
    layers.Bidirectional(layers.LSTM(32)),
    layers.Dense(num_words / 2, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid'),
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

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
    lambda logs: logs.get('acc') > 0.9 and logs.get('val_acc') > 0.9
)

history = model.fit(x_data, y_data, epochs=10, shuffle=True, validation_split=0.2, callbacks=[stop_training])

histories.append(history.history)


# History
def plot_history(history, metrics=('loss',)):
    mpplot.figure(figsize=(10, 6))
    epochs = range(len(history[metrics[0]]))
    for metric in metrics:
        mpplot.plot(epochs, history[metric], label=metric)
    mpplot.legend()


plot_history(history.history, ('acc', 'val_acc'))


# Save Model
model.save('model_q9_1.h5')