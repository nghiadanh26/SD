import numpy as np
import pandas as pd
import random
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import GRU
from keras.callbacks import ModelCheckpoint

data = pd.read_csv("LA_TRANSITION_ECOLOGIQUE.csv")
relevant_columns_id = [11, 13, 15, 16, 17, 19, 21, 24, 25] #23

questions = data[data.columns[relevant_columns_id]]
raw_text = ""
nbs_responses = []
for i in range(len(questions.columns)):
    k = 0
    print(i)
    for answer in questions[questions.columns[i]].dropna():
        if len(answer) <= 1000:
            continue
        k += 1
        raw_text += answer + " ||| "
    nbs_responses.append(k)
print(len(raw_text))
print(nbs_responses)

print("preprocessing of raw_text")
raw_text = raw_text.replace('.', ' . ')
raw_text = raw_text.replace('\n', ' \n ')
raw_text = raw_text.replace('\t', ' ')
raw_text = raw_text.replace('\u200e', 'r')
raw_text = raw_text.replace('\uf0f9', '')
raw_text = raw_text.replace('\u200b', '')
raw_text = raw_text.replace('\uf072', '')
raw_text = raw_text.replace('\uf0a7', '')
raw_text = raw_text.replace('\u2009', ' ')
raw_text = raw_text.replace('\u202f', ' ')
raw_text = raw_text.replace('\uf0e8', '')
raw_text = raw_text.replace('\u2003', '')
raw_text = raw_text.replace('\uf0b7', '')
raw_text = raw_text.replace('\uf0b8', '')
raw_text = raw_text.replace('\uf0e0', '')
raw_text = raw_text.replace('\uf04a', '')
raw_text = raw_text.replace('\xad', '')
raw_text = raw_text.replace('\uf0d8', '')
raw_text = raw_text.replace('\uf0fc', '')
raw_text = raw_text.replace('\xa0', '')
raw_text = raw_text.replace('\uf0f0', '')

char_set = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(char_set))
int_to_char = dict((i, c) for i, c in enumerate(char_set))

responses = raw_text.split(" ||| ")
random.shuffle(responses)

X = []
Y = []
seq_max = 100
nb_responses = 200

for response in responses[:nb_responses]:
    n = 0
    response = response + "|"
    for i in range(len(response)):
        l = [char_to_int["|"]]*seq_max
        for j in range(n):
            l[seq_max-j-1] = char_to_int[response[i-j-1]]

        if n < seq_max:
            n += 1

        X.append(l)
        Y.append(char_to_int[response[i]])

print("Sample size :", len(Y))

Y = keras.utils.to_categorical(Y, len(char_set))
X = np.reshape(X, (len(Y), seq_max, 1))/len(char_set)
print(X.shape, Y.shape)

print("END OF PREPROCESSING")
print("BUILIND MODEL")

# define the GRU model
model = Sequential()
model.add(GRU(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(GRU(256))
model.add(Dropout(0.2))
model.add(Dense(Y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
# define the checkpoint
filepath="weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

print("FIT")
# fit the model
model.fit(X, Y, epochs=50, batch_size=64, callbacks=callbacks_list)

print("1st TEST")

import sys
pattern = [char_to_int["|"]]*100
for i in range(1000):
    x = numpy.reshape(pattern, (1, len(pattern), 1))/len(char_set)
    prediction = model.predict(x, verbose=0)
    index = np.argmax(prediction)
    result = int_to_char[index]
    sys.stdout.write(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
    if result == "|":
        break
