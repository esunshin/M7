from learn import SEQUENCES_NAME, pickle, array, INPUT_LENGTH, reshape
from random import randint
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import sys

possible_chars = 'abcdefghijklmnopqrstuvwxyz \n'

def seed_is_legal(seed):
    for char in seed:
        if char not in possible_chars:
            return False
    return True

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

def print_error():
    print("Error – expecting:")
    print("`python3 create.py <'seedText'> <outputLength>`")
    print("or")
    print("`python3 create.py <'seedText'>`")
    print("or")
    print("`python3 create.py`")

def main():

    
    seed = ''
    user_seed = ''
    gen_length = '-1'

    if len(sys.argv) > 1:
        seed = sys.argv[1].strip().lower()
    if len(sys.argv) == 3:
        gen_length = sys.argv[2]
    elif len(sys.argv) > 3:
        print_error()
        return

    if seed != '':
        if seed == 'random':
            for _ in range(INPUT_LENGTH):
                user_seed += possible_chars[randint(0, len(possible_chars)-1)]
        elif seed_is_legal(seed):
            user_seed = seed[:INPUT_LENGTH]
            user_seed = (INPUT_LENGTH - len(user_seed)) * ' ' + user_seed
        else:
            sequences = pickle.load(open(SEQUENCES_NAME, "rb"))
            user_seed = sequences[randint(0, len(sequences))]
    else:
        sequences = pickle.load(open(SEQUENCES_NAME, "rb"))
        user_seed = sequences[randint(0, len(sequences))]

    if gen_length.isnumeric():
        gen_length = int(gen_length)
        if gen_length < 0:
            gen_length = 500
    else:
        gen_length = 500

    print(gen_length)

    print(user_seed + "\n")


    tokenizerPath = 'tokenizer2.p'
    modelFilePath = 'model.h5'


    tokenizer = pickle.load(open(tokenizerPath, "rb"))
    model = load_model(modelFilePath)
    seq = generate_seq(model, tokenizer, INPUT_LENGTH, seed, gen_length)


    print()
    print(seq)


if __name__ == "__main__":
    main()