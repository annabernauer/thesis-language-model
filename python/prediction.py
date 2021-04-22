
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from pickle import load


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



# load the model
model = load_model('model.h5')

# load the tokenizer
tokenizer = load(open('tokenizer.pkl', 'rb'))

seed_text = ("für ihre rasche rückmeldung bedanke ich mich",
             "für ihre schnelle rückmeldung bedanke ich mich",
             "für ihre sehr schnelle rückmeldung bedanke ich mich",
             "für Ihre Anfrage bedanke ich mich und",
             "hiermit bedanke ich mich herzlich für Ihre",
             "hiermit bedanke ich mich für Ihre Bestellung")


seed_text = [word.lower() for word in seed_text]

x_seq_length = len(seed_text[0].split(" ")) - 1

for seed in seed_text:
  print("##########################################################################################")
  print("seed: " + seed)
  print(generate_text_seq(model, tokenizer, x_seq_length, seed, 40))


