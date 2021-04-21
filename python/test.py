import re
import string
from tensorflow.keras.preprocessing.text import Tokenizer

def pad_punctuation(s): return re.sub(f"([{string.punctuation}])", r' \1 ', s)

S = ["The quick brown fox, jumped over the lazy dog."]
S = [pad_punctuation(s) for s in S]

t = Tokenizer(filters='')
t.fit_on_texts(S)
print(t.word_index)