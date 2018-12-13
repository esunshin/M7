# from scrape_standup import LOGUES_NAME, CHARS_NAME, pickle
import pickle
import re
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
from numpy import array, reshape
from keras.models import load_model

# import keras

SEQUENCES_NAME = "sequences.p"
LOGUES_NAME = 'inputMore.txt'
# monologues = pickle.load(open(LOGUES_NAME, "rb"))
# characters = pickle.load(open(CHARS_NAME, "rb"))

INPUT_LENGTH = 50

FILES = [
    'trevorNoah.txt',
    'sarahSilverman.txt',
    'johnMulaney.txt',
    'jerrySeinfeld.txt'
]

# PUNCT_REG = re.compile(r"""["\-:;,.?!()*]""")
PUNCT_REG = re.compile(r"""["\-:;,.?!()*0-9…$‘’“”_]""") # to remove numbers as well
APOST_REG = re.compile(r"'")
SPACE_REG = re.compile(r' +')

# print(monologues)


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

def segment_text():
    monologues = ""
    with open(LOGUES_NAME, "r") as f:
        monologues += f.read()

    # monologues = pickle.load(open(LOGUES_NAME, "rb"))
    tokens = [char for char in monologues]
    print("All tokens: " + str(len(tokens)))
    print("Unique: " + str(len(list(set(tokens)))))
    return tokens

def generate_sequences(tokens):
    sequence_length = INPUT_LENGTH + 1
    sequences = []
    for x in range(sequence_length, len(tokens)):
        sequences.append("".join(tokens[x - sequence_length: x]))
    pickle.dump(sequences, open(SEQUENCES_NAME, "wb"))
    print(str(len(sequences)) + " sequences")

def tokenize_sequences(sequences):
    tokenizer = Tokenizer(filters=None, char_level=True)
    tokenizer.fit_on_texts(sequences)
    return tokenizer, tokenizer.texts_to_sequences(sequences)

def split_input_output(sequences, vocab_size):
    tokenized_sequences = array(sequences)
    x, y = tokenized_sequences[:,:-1], tokenized_sequences[:,-1]
    y = to_categorical(y, num_classes=vocab_size)
    return x, y

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

def main2():
    segments = segment_text_restrictive()

def main3():

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


    pickle.dump(tokenizer, open('tokenizerNew.p', 'wb'))

    model = load_model('model.h5')

    checkpoint = ModelCheckpoint('tmp/newWeights.{epoch:02d}.hdf5')
    stop_early = EarlyStopping(monitor='loss', patience=2)

    callbacks = [checkpoint, stop_early]
    # fit model
    model.fit(x, y, batch_size=512, epochs=25, callbacks=callbacks)

    model.save('modelNew.h5')


if __name__ == "__main__":
    main()
    #main2()