"""
name: Ezra Sunshine
course: CSCI 3725
assignment: M7
date: 12/13/18
description: trains Keras model on Seinfeld stand up routines

some code/structure borrowed from:
 - https://www.analyticsvidhya.com/blog/2018/03/text-generation-using-python-nlp/
 - https://machinelearningmastery.com/how-to-develop-a-word-level-neural-language-model-in-keras/
"""

import pickle
import re
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
from numpy import array, reshape

SEQUENCES_NAME = "sequences.p"
LOGUES_NAME = 'routines/jerrySeinfeld.txt'

INPUT_LENGTH = 250

PUNCT_REG = re.compile(r"""["\-:;,.?!()*0-9…$‘’“”_]""") # to remove numbers as well
APOST_REG = re.compile(r"'")
SPACE_REG = re.compile(r' +')

"""
Cleans the input text, and breaks it into a list of characters
"""
def segment_text_restrictive():
    monologues = ""
    with open(LOGUES_NAME, "r") as f:
        monologues += f.read()

    monologues = APOST_REG.sub('', monologues)
    monologues = PUNCT_REG.sub(' ', monologues)
    monologues = SPACE_REG.sub(' ', monologues)

    monologues = monologues.lower()

    words = monologues.split(' ')
    print("All words: " + str(len(words)))
    print("Unique: " + str(len(list(set(words)))))
    print()

    tokens = [char for char in monologues]
    # print(tokens)
    print("All chars: " + str(len(tokens)))
    print("Unique: " + str(len(list(set(tokens)))))
    print(list(set(tokens)))

    return tokens

"""
Creates a list of strings that represent the training data.
Each string contains an (INPUT_LENGTH+1)-long series of characters
The last character of each string is the expected output, when the input is
 the rest of the string's characters
"""
def generate_sequences(tokens):
    sequence_length = INPUT_LENGTH + 1
    sequences = []
    for x in range(sequence_length, len(tokens)):
        sequences.append("".join(tokens[x - sequence_length: x]))
    pickle.dump(sequences, open(SEQUENCES_NAME, "wb"))
    print(str(len(sequences)) + " sequences")

"""
Trains a tokenizer on the input, which sets up the mapping from 
 character to integer for later one-hot encoding/decoding
Then created tokenized arrays from the string sequences
"""
def tokenize_sequences(sequences):
    tokenizer = Tokenizer(filters=None, char_level=True)
    tokenizer.fit_on_texts(sequences)
    return tokenizer, tokenizer.texts_to_sequences(sequences)

"""
Splits the tokenized arrays into an array of input arrays, and array of the output for each input array
"""
def split_input_output(sequences, vocab_size):
    tokenized_sequences = array(sequences)
    x, y = tokenized_sequences[:,:-1], tokenized_sequences[:,-1]
    y = to_categorical(y, num_classes=vocab_size)
    return x, y

"""
Creates the Keras model
"""
def create_model(vocab_input_dim, dense_output_dim, input_sequence_length):
    model = Sequential()
    model.add(Embedding(vocab_input_dim, dense_output_dim, input_length=input_sequence_length))
    model.add(LSTM(500, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(500))
    model.add(Dropout(0.2))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(vocab_input_dim, activation='softmax'))
    print(model.summary())
    return model

"""
Fits the model on the training data, and saves the tokenizer and model
"""
def fit_model_and_save(tokenizer, model, x_in, y_in, batch_size, epochs):
    pickle.dump(tokenizer, open('tokenizer.p', 'wb'))
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    checkpoint = ModelCheckpoint('tmp/weights.{epoch:02d}.hdf5')
    stop_early = EarlyStopping(monitor='loss', patience=2)

    callbacks = [checkpoint, stop_early]
    # fit model
    model.fit(x_in, y_in, batch_size=batch_size, epochs=epochs, callbacks=callbacks)
    # save the model to file
    model.save('model.h5')
    # save the tokenizer
    pickle.dump(tokenizer, open('tokenizer2.p', 'wb'))

"""
Manages cleaning/formating the input data and the training the model on it.
"""
def main():
    segments = segment_text_restrictive() 
    # segments = segment_text()
    generate_sequences(segments)
    sequences = pickle.load(open(SEQUENCES_NAME, "rb"))


    tokenizer, tokenized_sequences = tokenize_sequences(sequences)
    vocab_size = len(tokenizer.word_index) + 1
    print(vocab_size)

    # separate into input and output
    x, y = split_input_output(tokenized_sequences, vocab_size)
    seq_length = x.shape[1]
    print(x)
    print(y)

    # define model
    print("vocab_size: ", vocab_size, " seq_length: ", seq_length)

    model = create_model(vocab_size, 16, seq_length)
    
    fit_model_and_save(tokenizer, model, x, y, 512, 50)

"""
For testing purposes
"""
def main2():
    segments = segment_text_restrictive()

if __name__ == "__main__":
    main()
    #main2()
