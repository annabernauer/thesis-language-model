
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from pickle import load
from pathlib import Path


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


n = 15

# load the model
model = load_model('model.h5')

# load the tokenizer
tokenizer = load(open('tokenizer.pkl', 'rb'))

seed_text = ("Für Ihre rasche Rückmeldung bedanke ich mich herzlich und darf Ihnen anbei unser neues Angebot für Ihre Pins senden . ",
            "Für Ihre schnelle Rückmeldung bedanke ich mich herzlich und darf Ihnen anbei unser neues Angebot für Ihre Pins senden .",
             "Sehr gerne sende ich Ihnen vorab und kostenlos einen Gestaltungsvorschlag für Ihre Pins . Bei Interesse bitte ich hier",
            "gerne sende ich Ihnen vorab und kostenlos einen Gestaltungsvorschlag für Ihre Pins . Bei Interesse bitte ich hier um"
            )

# seed_text = [seed.split() for seed in seed_text]
# seed_text = [seed[:n] for seed in seed_text]
# seed_text = [" ".join(seed) for seed in seed_text]
seed_text = [seed.lower() for seed in seed_text]

x_seq_length = len(seed_text[0].split(" ")) - 1

generated_texts = []

for seed in seed_text:
  generated_text = generate_text_seq(model, tokenizer, x_seq_length, seed, 40)
  generated_texts.append((seed, generated_text))

print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
[print(text) for text in generated_texts]

Path("target/").mkdir(parents=True, exist_ok=True)

f = open("target/generated_texts.txt", "w")
for text in generated_texts:
    f.write(f"<SEED> {text[0]} <GENERATED> {text[1]}\n")
f.close()

