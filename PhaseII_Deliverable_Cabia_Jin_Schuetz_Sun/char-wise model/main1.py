"""
    Inspired by https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
"""
from __future__ import print_function
#from numpy.random import seed
#seed(4185265454)
import numpy
#seed = numpy.random.get_state()
#print(seed[1][0])
from keras.callbacks import ModelCheckpoint
import helper 
import numpy as np
"""
    Define global variables.
"""
SEQUENCE_LENGTH = 40
SEQUENCE_STEP = 3
PATH_TO_CORPUS = "corpus.txt"
EPOCHS = 100
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

"""
    Define the structure of the model.
"""
model = helper.build_model(SEQUENCE_LENGTH, chars)
model.summary()

"""
    Train the model
"""

filepath = "model/model-{epoch:02d}-{val_loss:.4f}.h5"
#checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
#checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1)
checkpoint = ModelCheckpoint(filepath, monitor='val_loss',save_best_only=True, mode="min")
callbacks_list = [checkpoint]
hist = model.fit(X, y, batch_size=128, epochs=EPOCHS, verbose=1,  callbacks=callbacks_list, validation_split=0.25, shuffle=True)
np.save('model/hist.npy',hist.history)
#hist.history
## result:
#{'acc': [0.9234666666348775, 0.9744000000317892, 0.9805999999682109],
# 'loss': [0.249011807457606, 0.08651042315363884, 0.06568188704450925],
# 'val_acc': [0.9799, 0.9843, 0.9876],
# 'val_loss': [0.06219216037504375, 0.04431889447008725, 0.03649089169385843]}
