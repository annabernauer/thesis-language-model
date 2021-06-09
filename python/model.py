# %tensorflow_version 2.x
import logging
from pathlib import Path

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
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
import sys, os


class Tee(object):
  def __init__(self, *files):
    self.files = files

  def write(self, obj):
    for f in self.files:
      f.write(obj)
      f.flush()  # If you want the output to be visible immediately

  def flush(self):
    for f in self.files:
      f.flush()

########################################### pre-process text ###########################################

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

delimiter = "%&§§&%"
n = 5
epochs = 40 #30 old value
embeddings = 100
foldCount = 10
src_name = "messagesSmall"

logging.info(f"n = {n}, epochs = {epochs}, embeddings = {embeddings}, src_name = {src_name}, foldCount = {foldCount}")

# punctuation = r"""!"#$%&'()*+,-./:;=?@[\]^`{|}~"""
punctuation = r"""!?.,:;"""

def pad_punctuation(s): return re.sub(f"([{punctuation}])", r' \1 ', s)

dirName = f"{src_name}_n_{n}"

targetDir = f"target/{dirName}_emb_{embeddings}_epo_{epochs}"
logFile = f"{targetDir}/log.txt"

Path(targetDir).mkdir(parents=True, exist_ok=True)

f = open(logFile, 'w')
original = sys.stdout
sys.stdout = Tee(sys.stdout, f)

for fold in range(foldCount):
  logging.info(f"+++++++++++++++++++++++++++++ fold = {fold} +++++++++++++++++++++++++++++")
  foldDir = f"{dirName}_fold_{fold}"
  with open(f'resources/{dirName}/{foldDir}/historiesAndSequences/sequencesKeys.txt', encoding='utf-8') as f:
    lines = [line.rstrip() for line in f]
  lines = [line.strip() for line in lines]
  lines = [line.split(delimiter) for line in lines]
  lines = [x for x in lines if len(x) == n]

  lines = [" ".join(line) for line in lines]

  ########################################### create layers ###########################################

  lines = [pad_punctuation(s) for s in lines]
  # [print(f"{len(x)} {x}") for x in lines]

  tokenizer = Tokenizer(filters='')
  tokenizer.fit_on_texts(lines)
  sequences = tokenizer.texts_to_sequences(lines)                         #transforms each line of words to a sequence of integers

  #[print(word) for word in seed_text]

  # [print(len(x)) for x in sequences]
  # tokenizer.sequences_to_texts()
  sequences = [x for x in sequences if len(x) == n]

  sequences = np.array(sequences)
  X, y = sequences[:, :-1], sequences[:, -1]                              #split lines in the first 50 words in x and the last word in y

  # [print(f"{x}") for x in sequences]

  vocab_size = len(tokenizer.word_index) + 1

  # y = to_categorical(y, vocab_size)                                       #returns a binary matrix representation of the input
  seq_length = X.shape[1]                                                 #50

  # print(vocab_size)

  model = Sequential()
  model.add(Embedding(vocab_size, embeddings, input_length=seq_length))
  model.add(LSTM(100, return_sequences=True))
  model.add(LSTM(100))
  model.add(Dense(100, activation='relu'))
  model.add(Dense(vocab_size, activation='softmax'))

  #print(model.summary())

  # model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
  model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer='adam', metrics=['accuracy'])

  model.fit(X, y, batch_size = 256, epochs = epochs)

  targetFoldDir = f"{targetDir}/{foldDir}"
  # save the model to file
  model.save(f'{targetFoldDir}/model.h5')
  # save the tokenizer
  dump(tokenizer, open(f'{targetFoldDir}/tokenizer.pkl', 'wb'))

  logging.info(f"fold {fold} finished, files are saved")

# sys.stdout.close()
f.close()

