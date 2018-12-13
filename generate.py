from learn import SEQUENCES_NAME, pickle, array, INPUT_LENGTH, reshape
from random import randint
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences


# generate a sequence from a language model
def generate_seq(model, tokenizer, seq_length, seed_text, n_words):
    result = list()
    in_text = seed_text
    # generate a fixed number of words
    for _ in range(n_words):
        # encode the text as integer
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        # print(in_text, encoded)

        # truncate sequences to a fixed length
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
        # predict probabilities for each word
        yhat = model.predict_classes(encoded, verbose=0)
        # map predicted word index to word
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == yhat:
                out_word = word
                break
        # append to input
        in_text += '' + out_word
        result.append(out_word)
    return ''.join(result)

def generate_seq_no_embedding(model, tokenizer, seq_length, seed_text, n_words):
    in_text = seed_text
    for _ in range(n_words):
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        # print(encoded)
        # print()
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
        # print(encoded)
        # print()
        x = reshape(encoded, (1, len(encoded[0]), 1))
        # print(x)
        # print()
        # x = x / float(len(seq_length))
        x = x / float(seq_length)
        # print()
        yhat = model.predict_classes(x, verbose=0)
        # print(yhat)
        out_char = ''
        for char, index in tokenizer.word_index.items():
            if index == yhat:
                out_char = char
                break
        in_text += out_char
    return in_text

def main():
    sequences = pickle.load(open(SEQUENCES_NAME, "rb"))
    seed = sequences[randint(0, len(sequences))]
    print(seed + "\n")

    tokenizerPath = 'longRun/tokenizer.p'
    modelFilePath = 'longRun/model.h5'


    tokenizer = pickle.load(open(tokenizerPath, "rb"))
    model = load_model(modelFilePath)
    seq = generate_seq(model, tokenizer, INPUT_LENGTH, seed, 100)

    # tokenizer = pickle.load(open('tokenizerLSTM.p', "rb"))
    # model = load_model('modelLSTM.h5')
    # seq = generate_seq_no_embedding(model, tokenizer, INPUT_LENGTH, seed, 200)

    print()
    print(seq)


if __name__ == "__main__":
    main()