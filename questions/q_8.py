# imdb, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)


import tensorflow as tf
from tensorflow.keras import layers, regularizers
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


max_length = 100
x_train, num_words, tokenizer = texts_to_sequences(texts_train, max_length)
x_valid, _, _ = texts_to_sequences(texts_valid, max_length, tokenizer)
y_train = np.array(labels_train)
y_valid = np.array(labels_valid)


# Model
embedding_dim = 100
model = tf.keras.models.Sequential([
    layers.Embedding(num_words, embedding_dim),
    layers.Conv1D(16, 5, activation='relu'),
    layers.MaxPooling1D(4),
    layers.Bidirectional(layers.LSTM(16, return_sequences=True)),
    layers.Bidirectional(layers.LSTM(16)),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
histories = []


# Train
epochs = 10
stop_training = callback_of_stop_training(lambda logs: logs.get('acc') > 0.8 and logs.get('val_acc') > 0.8)
history = model.fit(
    x_train, y_train, validation_data=(x_valid, y_valid), epochs=epochs,
    callbacks=[stop_training]
)
histories.append(history.history)
plot_history(history.history, ('acc', 'val_acc'))


# History
saved_history = flat_histories(histories)
plot_history(saved_history, ('acc', 'val_acc'))

plot([saved_history['acc'][:epochs], history.history['acc']], ['old', 'new'])
plot([saved_history['val_acc'][:epochs], history.history['val_acc']], ['old', 'new'])


# Save Model
model.save('model_q8_1.h5')

