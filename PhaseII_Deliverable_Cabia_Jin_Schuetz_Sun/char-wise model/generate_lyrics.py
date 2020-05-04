from __future__ import print_function

import numpy as np
import sys
from keras.models import load_model
import helper 
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="This script trains the LSTM model to generate new lyrics",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--start_40", type=str, default='I loved how you walk with your hands in ',
                        help="Please input 40 characters including whitespace as the start sentence of a new song~")
    parser.add_argument("--load_model", type=str, default='model_best/model-20-1.4628.h5',
                        help="Please select the model you will use")    
    args = parser.parse_args()
    return args


"""
    Define global variables.
"""
SEQUENCE_LENGTH = 40
SEQUENCE_STEP = 3
PATH_TO_CORPUS = "corpus.txt"
EPOCHS = 20
DIVERSITY = 1.0

"""
    Read the corpus and get unique characters from the corpus.
"""
text = helper.read_corpus(PATH_TO_CORPUS)
chars = helper.extract_characters(text)

"""
    Create sequences that will be used as the input to the network.
    Create next_chars array that will serve as the labels during the training.
"""
sequences, next_chars = helper.create_sequences(text, SEQUENCE_LENGTH, SEQUENCE_STEP)
char_to_index, indices_char = helper.get_chars_index_dicts(chars)

"""
    The network is not able to work with characters and strings, we need to vectorise.
"""
X, y = helper.vectorize(sequences, SEQUENCE_LENGTH, chars, char_to_index, next_chars)

m = get_args().load_model
model = load_model(m)

for diversity in [0.2, 0.5, 1.0, 1.2]:
    print()
    print('----- diversity:', diversity)

    generated = ''
    # insert your 40-chars long string. OBS it needs to be exactly 40 chars!
    sentence = get_args().start_40
    sentence = sentence.lower()
    if len(sentence)<40:
           n = 40-len(sentence)
           sentence = ' '*n+sentence
    if len(sentence)>40:
           sentence = sentence[-40:]
    generated += sentence

    print('----- Generating with seed: "' + sentence + '"')
    sys.stdout.write(generated)

    for i in range(400):
        x = np.zeros((1, SEQUENCE_LENGTH, len(chars)))
        for t, char in enumerate(sentence):
            x[0, t, char_to_index[char]] = 1.

        predictions = model.predict(x, verbose=0)[0]
        next_index = helper.sample(predictions, diversity)
        next_char = indices_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

        sys.stdout.write(next_char)
        sys.stdout.flush()
    print()



