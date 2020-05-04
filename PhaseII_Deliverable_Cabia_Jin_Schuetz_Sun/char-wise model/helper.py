"""

    Utils functions for LSTM network.

"""

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import LSTM
from keras.optimizers import RMSprop
import io
import numpy as np
import re
import string

def create_sequences(text, sequence_length, step):
    sequences = []
    next_chars = []
    for i in range(0, len(text) - sequence_length, step):
        sequences.append(text[i: i + sequence_length])
        next_chars.append(text[i + sequence_length])
    return sequences, next_chars


def build_model(sequence_length, chars):
    model = Sequential()
    model.add(LSTM(64, input_shape=(sequence_length, len(chars))))
#    model.add(LSTM(16, input_shape=(sequence_length, len(chars)),return_sequences=True))
#    model.add(LSTM(16 ,return_sequences=True))
    model.add(Dropout(0.3))
#    model.add(Flatten())
#    print(len(chars))
    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))
    learning_rate = 0.01
    decay_rate = learning_rate / 100
    optimizer = RMSprop(lr=learning_rate, clipnorm=1., decay=decay_rate)
#    optimizer = RMSprop(lr=learning_rate, clipnorm=1.)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,metrics=['accuracy'])
    return model


def sample(preds, temperature=1.0):

    if temperature == 0:
        temperature = 1

    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def extract_characters(text):
    return sorted(list(set(text)))


def get_chars_index_dicts(chars):
    return dict((c, i) for i, c in enumerate(chars)), dict((i, c) for i, c in enumerate(chars))

#def remove_special_characters(text, remove_digits=True):
#    pattern = r'[^a-zA-z0-9\s]+' if not remove_digits else r'[^a-zA-z\s]+'
#    text = re.sub(pattern, '', text)
#    return text

def read_corpus(path):
    with io.open(path, 'r', encoding='utf8') as f:
#           print(len(f.read().lower()))
#           text = remove_special_characters('&djjljijeijijj*_  998sq')
           doc = ''.join(c for c in f.read().lower() if c in string.ascii_lowercase+' ')
           doc = re.sub(' +', ' ', doc)
           return doc 
        


def vectorize(sequences, sequence_length, chars, char_to_index, next_chars):
    X = np.zeros((len(sequences), sequence_length, len(chars)), dtype=np.bool)
    y = np.zeros((len(sequences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sequences):
        for t, char in enumerate(sentence):
            X[i, t, char_to_index[char]] = 1
        y[i, char_to_index[next_chars[i]]] = 1

    return X, y

