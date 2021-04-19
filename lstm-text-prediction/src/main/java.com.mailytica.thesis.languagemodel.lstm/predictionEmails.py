# %tensorflow_version 2.x
import tensorflow as tf
import string
import requests


########################################### pre-process text ###########################################

# def clean_text(doc):
#   tokens = doc.split()
#   table = str.maketrans('', '', string.punctuation)
#   tokens = [w.translate(table) for w in tokens]
#   tokens = [word for word in tokens if word.isalpha()]
#   tokens = [word.lower() for word in tokens]
#   return tokens
#
#
# response = requests.get('https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt')
# data = response.text.split('\n')
# data = data[253:]
# data = " ".join(data)
# tokens = clean_text(data)
#
#
# length = 2 + 1
# lines = []
#
# for i in range(length, len(tokens)):
#   seq = tokens[i-length:i]
#   line = ' '.join(seq)
#   lines.append(line)
#   if i > 100:
#     break


delimiter = "%&§§&%"


def pad_punctuation(s): return re.sub(f"([{string.punctuation}])", r' \1 ', s)

with open('sequencesKeys.txt', encoding='utf-8') as f:
    lines = [line.rstrip() for line in f]
lines = [line.strip() for line in lines]
lines = [line.split(delimiter) for line in lines]
lines = [x for x in lines if len(x) == 3]
# [print(f"{len(x)} {x}") for x in lines]
lines = [" ".join(line) for line in lines]


# with open('sequencesKeys.txt', encoding='utf-8') as f:
#   lines = [line.rstrip() for line in f]
# lines = [line.strip() for line in lines]
# lines = [" ".join(line.split(delimiter)) for line in lines]


########################################### create layers ###########################################

import numpy as np
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences


lines = [pad_punctuation(s) for s in lines]
[print(f"{len(x)} {x}") for x in lines]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
sequences = tokenizer.texts_to_sequences(lines)                         #transforms each text in texts to a sequence of integers

# [print(len(x)) for x in sequences]
# tokenizer.sequences_to_texts()
sequences = [x for x in sequences if len(x) == 3]

sequences = np.array(sequences)
X, y = sequences[:, :-1], sequences[:, -1]                              #split lines in the first 50 words in x and the last word in y

vocab_size = len(tokenizer.word_index) + 1

y = to_categorical(y, vocab_size)                                       #returns a binary matrix representation of the input

seq_length = X.shape[1]                                                 #50

print(seq_length)
print(vocab_size)

model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=seq_length))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100))
model.add(Dense(100, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))

#print(model.summary())

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

model.fit(X, y, batch_size = 256, epochs = 10)

def generate_text_seq(model, tokenizer, text_seq_length, seed_text, n_words):
    text = []

    for _ in range(n_words):
        encoded = tokenizer.texts_to_sequences([seed_text])[0]
        encoded = pad_sequences([encoded], maxlen = text_seq_length, truncating='pre')

        y_predict = model.predict_classes(encoded)

        predicted_word = ''
        for word, index in tokenizer.word_index.items():
            if index == y_predict:
                predicted_word = word
                break
        seed_text = seed_text + ' ' + predicted_word
        text.append(predicted_word)
    return ' '.join(text)


seed_text = lines[123]
print(len(lines))

print(seed_text)

print(generate_text_seq(model, tokenizer, seq_length, seed_text, 100))


