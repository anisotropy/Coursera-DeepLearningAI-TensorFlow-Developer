# imdb_reviews/subwords8k

import tensorflow as tf
from tensorflow.keras import layers, Input, regularizers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow_datasets as tfds
import numpy as np
import tensorflow_datasets as tfds


# Get Data
import tensorflow_datasets as tfds
imdb, info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)


# Prepare Data
def padded_batch_ds(raw_ds, batch_size, shuffle_buffer_size=None):
    if shuffle_buffer_size is None:
        shuffled_ds = raw_ds
    else:
        shuffled_ds = raw_ds.shuffle(shuffle_buffer_size)
    ds = shuffled_ds.padded_batch(batch_size, tf.compat.v1.data.get_output_shapes(raw_ds))
    return ds


tokenizer = info.features['text'].encoder

num_words = tokenizer.vocab_size
ds_train = padded_batch_ds(imdb['train'], batch_size=32, shuffle_buffer_size=1000)
ds_valid = padded_batch_ds(imdb['test'], batch_size=32)


# Model
embedding_dim = 64
model = tf.keras.models.Sequential([
    Input(shape=(None,)),
    layers.Embedding(num_words, embedding_dim),
    layers.Conv1D(64, 5, activation='relu'),
    layers.MaxPooling1D(4),
    layers.Bidirectional(layers.LSTM(32, return_sequences=True)),
    layers.Bidirectional(layers.LSTM(32)),
    layers.Dense(num_words / 2, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid'),
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
history = model.fit(
    ds_train,
    validation_data=ds_train,
    epochs=10,
    callbacks=[stop_training]
)

histories.append(history.history)


# Save Model
model.save('model_q10_1.h5')