# %tensorflow_version 2.x
import tensorflow as tf
import string
import requests
import numpy as np
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pickle import dump

########################################### pre-process text ###########################################

delimiter = "%&§§&%"
def pad_punctuation(s): return re.sub(f"([{string.punctuation}])", r' \1 ', s)

with open('sequencesKeys.txt', encoding='utf-8') as f:
  lines = [line.rstrip() for line in f]
lines = [line.strip() for line in lines]
lines = [line.split(delimiter) for line in lines]
lines = [x for x in lines if len(x) == 7]
# [print(f"{len(x)} {x}") for x in lines]
lines = [" ".join(line) for line in lines]

########################################### create layers ###########################################

lines = [pad_punctuation(s) for s in lines]
# [print(f"{len(x)} {x}") for x in lines]

tokenizer = Tokenizer(filters='')
tokenizer.fit_on_texts(lines)
sequences = tokenizer.texts_to_sequences(lines)                         #transforms each text in texts to a sequence of integers


#[print(word) for word in seed_text]

# [print(len(x)) for x in sequences]
# tokenizer.sequences_to_texts()
sequences = [x for x in sequences if len(x) == 7]

sequences = np.array(sequences)
X, y = sequences[:, :-1], sequences[:, -1]                              #split lines in the first 50 words in x and the last word in y

# [print(f"{x}") for x in sequences]

vocab_size = len(tokenizer.word_index) + 1

y = to_categorical(y, vocab_size)                                       #returns a binary matrix representation of the input

seq_length = X.shape[1]                                                 #50

print(seq_length)
# print(vocab_size)

model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=seq_length))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100))
model.add(Dense(100, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))

#print(model.summary())

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

model.fit(X, y, batch_size = 256, epochs = 100)

# save the model to file
model.save('model.h5')
# save the tokenizer
dump(tokenizer, open('tokenizer.pkl', 'wb'))


