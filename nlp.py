import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow_datasets as tfds
import numpy as np


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
            input_seqs.append(seq[:i + 1])

    padded_seqs = pad_sequences(input_seqs, maxlen=max_length)

    x_train = padded_seqs[:, :-1]
    y_train = tf.keras.utils.to_categorical(padded_seqs[:, -1], num_classes=num_words)

    sequence_length = max_length - 1

    return x_train, y_train, num_words, sequence_length, tokenizer


def embedding_matrix_from_file(filepath, tokenizer, num_words):
    embedding_matrix = None
    embedding_dim = None
    with open(filepath, 'r') as file:
        for line in file:
            row = line.split()
            word = row[0]
            vector = np.array(row[1:], dtype='float')
            token = tokenizer.word_index.get(word)
            if embedding_matrix is None:
                embedding_dim = len(vector)
                embedding_matrix = np.zeros((num_words, embedding_dim))
            if token:
                embedding_matrix[token] = vector

    return embedding_matrix, embedding_dim


def pretrained_embedding_layer(num_words, embedding_dim, embedding_matrix):
    return layers.Embedding(
        num_words, embedding_dim,
        weights=[embedding_matrix], trainable=False
    )


def layers_0(num_words, embedding_dim):
    return [
        layers.Embedding(num_words, embedding_dim),
        layers.Conv1D(64, 5, activation='relu'),
        layers.MaxPooling1D(4),
        layers.Bidirectional(layers.LSTM(32, return_sequences=True)),
        layers.Bidirectional(layers.LSTM(32)),
        layers.Dense(num_words / 2, activation='relu'),
        layers.Dropout(0.5),
    ]


def layers_1(num_words, embedding_dim):
    return [
        layers.Embedding(num_words, embedding_dim),
        layers.GlobalAveragePooling1D(),
        layers.Dense(16, activation='relu'),
    ]


def layers_2(num_words, embedding_dim):
    return [
        layers.Embedding(num_words, embedding_dim),
        layers.Conv1D(128, 5, activation='relu'),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation='relu')
    ]


def layers_3(num_words, embedding_dim):
    return [
        layers.Embedding(num_words, embedding_dim),
        layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
        layers.Bidirectional(layers.LSTM(32)),
        layers.Dense(64, activation='relu'),
    ]


def layers_4(num_words, embedding_dim):
    return [
        layers.Embedding(input_dim=num_words, output_dim=embedding_dim),
        layers.Bidirectional(layers.GRU(32, return_sequences=True)),
        layers.Bidirectional(layers.GRU(32)),
        layers.Dense(6, activation='relu'),
    ]


def layers_5(num_words, embedding_dim):
    return [
        layers.Embedding(input_dim=num_words, output_dim=embedding_dim),
        layers.GlobalAveragePooling1D(),
        layers.Dense(6, activation='relu'),

    ]


def layers_6(num_words, embedding_dim):
    return [
        layers.Embedding(num_words, embedding_dim),
        layers.Bidirectional(layers.LSTM(150, return_sequences=True)),
        layers.Dropout(0.5),
        layers.Bidirectional(layers.LSTM(100)),
        layers.Dense(num_words / 2, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    ]


def layers_7(num_words, embedding_dim):
    return [
        layers.Embedding(num_words, embedding_dim),
        layers.Bidirectional(layers.LSTM(32)),
    ]


def layers_8(num_words, embedding_dim):
    return [
        layers.Embedding(num_words, embedding_dim),
        layers.Dropout(0.2),
        layers.Conv1D(64, 5, activation='relu'),
        layers.MaxPooling1D(4),
        layers.LSTM(64),
    ]


def predict_next_words(model, tokenizer, sequence_length, seed_text, num_next_words):
    new_text = seed_text
    seed_seqs = tokenizer.texts_to_sequences([new_text])

    reverse_word_index = {token: word for (word, token) in tokenizer.word_index.items()}

    for _ in range(num_next_words):
        seed_padded_seqs = pad_sequences(seed_seqs, maxlen=sequence_length)
        result = model.predict(seed_padded_seqs)
        next_token = np.argmax(result)
        seed_seqs = np.append(seed_padded_seqs[0], next_token)[np.newaxis]
        next_word = reverse_word_index.get(next_token)
        if next_word:
            new_text += ' ' + next_word

    return new_text
