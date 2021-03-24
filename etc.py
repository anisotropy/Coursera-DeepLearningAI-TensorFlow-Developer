import tensorflow as tf
import numpy as np
import matplotlib.pyplot as mpplot
import json
import os
import random
import zipfile


def clear_session():
    tf.keras.backend.clear_session()
    tf.random.set_seed(51)
    np.random.seed(51)


def fetch_from_url(url, filepath):
    tf.keras.utils.get_file(os.path.abspath(filepath), origin=url)


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


def save_model_weights(model, filepath):
    model.save_weights(filepath)


def load_model_weights(model, filepath):
    model.load_weights(filepath)


def save_histories(histories, filepath):
    with open(filepath, 'w') as file:
        json.dump(histories, file)


def load_histories(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)


def remove_stopwords(text, stopwords):
    words_without_stopwords = []
    for word in text.split():
        if not (word in stopwords):
            words_without_stopwords.append(word)
    return ' '.join(words_without_stopwords)


def trend(time, slope=0.0):
    return slope * time


def seasonal_pattern(season_time):
    return np.where(
        season_time < 0.1,
        np.cos(2 * season_time * np.pi),
        1 / np.exp(3 * season_time)
    )


def seasonality(time, period, amplitude=1, phase=0):
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)


def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level


def shuffle(seq):
    random.shuffle(seq)


def extract_zip(filename, to_dir):
    zip_ref = zipfile.ZipFile(filename, 'r')
    zip_ref.extractall(to_dir)
    zip_ref.close()

