import sys
import csv
import codecs
import numpy as np
from keras.models import Sequential
from keras.layers import Dropout, Dense, LSTM, GRU, Activation
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam, RMSprop
from make_parallel import make_parallel

####### Pre-processing the data #######
print "LSTM 2 X (256 added Dropout 0.5) because network was overfitting"
print "Reduced learning rate because loss was exploding towards the end"

# Reading the trump dataset into raw_text
f = codecs.open('data/trump_tweets.csv', 'r', encoding='ascii', errors='ignore')
data = csv.reader(f)
tweets = np.array([row[0].lower() for row in data])
tweets = tweets[:100]

temp = []
for i, t in enumerate(tweets):
    if len(t) != 0 and t[0] != '@':
        temp.append(t)

tweets = np.array(temp)

raw_text = ''
for t in tweets:
    raw_text += t.lower()

# set(raw_text) : set of all characters in the text (string)
# list(set(raw_text)) : converting the set to a list
# sorted(list(set(raw_text))) : sorting the list of characters
# storing the list of characters a dictionary with each character corresponding to an index
chars = sorted(list(set(raw_text)))
# chars.pop(64)
# chars.pop(64)
# chars.pop(64)
# chars.insert(35, '\\')
# chars.pop(38)
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

# Saving the dictionaries as npy files 
np.save('trump_char_to_int.npy', char_to_int)
np.save('trump_int_to_char.npy', int_to_char)

n_chars = len(raw_text)
n_vocab = len(chars)
print "Total chars in text: ", n_chars
print "Total chars in vocabulary: ", n_vocab

seq_len = 50
# skipping 3 step chars in between two examples
step = 3
X = []
y = []
for i in range(0, n_chars - seq_len, step):
    X.append(raw_text[i:i + seq_len])
    y.append(raw_text[i + seq_len])
train_size = len(X)

print "Number of training examples: ", train_size

# we reshape X to [num_samples, length_sample, features] in order to pass it to the keras layer
# next we store the characters as one hot encoding
# convert y also to one hot vector
X_train = np.zeros((train_size, seq_len, n_vocab), dtype=np.bool)
y_train = np.zeros((train_size, n_vocab), dtype=np.bool)
for i, sentence in enumerate(X):
    for j, ch in enumerate(sentence):
        if ch in char_to_int:
            X_train[i, j, char_to_int[ch]] = 1
    if y[i] in char_to_int:
        y_train[i, char_to_int[y[i]]] = 1

# Setting up the model

model = Sequential()
model.add(LSTM(256, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True, unroll=False))
model.add(Dropout(0.5))
model.add(LSTM(256))
model.add(Dropout(0.5))
model.add(Dense(y_train.shape[1]))
model.add(Activation('softmax'))
rmsprop = RMSprop(lr=0.001, clipvalue=5, decay=0.)
# model.load_weights("weights/weights-improvement-LSTM256-00-1.9579.hdf5")
model = make_parallel(model, 2)
model.compile(optimizer=rmsprop, loss='categorical_crossentropy', metrics=['accuracy'])

# Creating the checkpoint saving functionality due to computational constraints

'''
path = "weights/weights-improvement-trump-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(path, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]


def gen_index(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


print
####### Fitting the model with the data #######
for iter in range(1, 150):

    print "Iteration: ", iter
    model.fit(X_train, y_train, batch_size=64, nb_epoch=1, callbacks=callbacks_list, validation_split=0.2)

    start = np.random.randint(0, len(raw_text) - seq_len - 1)

    generated = ''
    sentence = raw_text[start: start + seq_len]
    generated += sentence
    print "Seed: ", sentence

    cnt = 0
    while cnt < 40:
        x = np.zeros((1, seq_len, n_vocab))

        for t, ch in enumerate(sentence):
            if ch in char_to_int:
                x[0, t, char_to_int[ch]] = 1
            else:
                x[0, t, char_to_int[' ']] = 1

        pred = model.predict(x, verbose=0)[0]
        temperature = 0.5
        next_index = int(gen_index(pred, temperature))
        next_char = int_to_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

        # print next_char
        sys.stdout.write(next_char)
        sys.stdout.flush()

        cnt += 1
        if next_char in ('.', '?', '!') and cnt > 40:
            break
    print
'''
