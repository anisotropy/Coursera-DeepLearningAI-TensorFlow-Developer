# imdb_reviews

import tensorflow as tf
from tensorflow.keras import layers, Input, regularizers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow_datasets as tfds
import numpy as np
import tensorflow_datasets as tfds
import matplotlib.pyplot as mpplot


# Get Data
def load_imdb():
    imdb, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)
    raw_ds_train, row_ds_valid = imdb['train'], imdb['test']

    texts_train = []
    labels_train = []
    texts_valid = []
    labels_valid = []
    for s, l in raw_ds_train:
        texts_train.append(str(s.numpy()))
        labels_train.append(l.numpy())
    for s, l in raw_ds_train:
        texts_valid.append(str(s.numpy()))
        labels_valid.append(l.numpy())

    return texts_train, labels_train, texts_valid, labels_valid


texts_train, labels_train, texts_valid, labels_valid = load_imdb()


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


max_length = 500
x_train, num_words, tokenizer = texts_to_sequences(texts_train, max_length)
x_valid, _, _ = texts_to_sequences(texts_valid, max_length, tokenizer)
y_train = np.array(labels_train)
y_valid = np.array(labels_valid)


# Model
embedding_dim = 64
model = tf.keras.models.Sequential([
    Input(shape=(max_length,)),
    layers.Embedding(num_words, embedding_dim),
    layers.Conv1D(64, 5, activation='relu'),
    layers.MaxPooling1D(4),
    layers.Bidirectional(layers.LSTM(32, return_sequences=True)),
    layers.Bidirectional(layers.LSTM(32)),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1)
])
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
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
    lambda logs: logs.get('acc') > 0.9 and logs.get('val_acc') > 0.9
)
history = model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=10, callbacks=[stop_training])
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
model.save('model_q8_1.h5')
