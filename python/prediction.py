
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from pickle import load
from pathlib import Path
from pyspark.sql import *

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

n = 5
epochs = 20  #30 old value
embeddings = 100
src_name = "messagesSmall"
foldCount = 10

srcName = f"{src_name}_n_{n}"

for fold in range(foldCount):
  foldDir = f"{srcName}_fold_{fold}"
  targetFoldDir = f"target/{srcName}_emb_{embeddings}_epo_{epochs}/{foldDir}"

  # load the model
  model = load_model(f'{targetFoldDir}/model.h5')

  # load the tokenizer
  tokenizer = load(open(f'{targetFoldDir}/tokenizer.pkl', 'rb'))

  spark = SparkSession \
    .builder \
    .appName("Python Spark SQL basic example") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

  df = spark.read.csv(f'resources/{srcName}/{foldDir}/testData/part-00000-fac4cd2f-9089-4789-90bf-147327fcf314-c000.csv', header="true", inferSchema="true")

  # seeds = ("Mögliche Änderungswünsche nehmen wir sehr",
  # "Sollten Sie weitere Fragen haben",
  # "Alternativ würde ich mich jederzeit")

  # result_lst = df.rdd.map(lambda row: row.getString(0)).collect()
  # [print(x) for x in result_lst]

  seeds = [str(row.seeds) for row in df.select("seeds").collect()]
  reference = [str(row.referenceSentences) for row in df.select("referenceSentences").collect()]

  seedsWithReference = list(zip(seeds, reference))
  [print(x) for x in seedsWithReference]

# df.rdd.map()
# val result = df.rdd.map(row => (row.getDouble(0), row.getDouble(1))).collect()
#   result = df.rdd.map(row=> (row.getDouble(0), row.getDouble(1))).collect()


  # [print(f"{x}") for x in seeds]

  # seed_text = [seed.split() for seed in seed_text]
  # seed_text = [seed[:n] for seed in seed_text]
  # seed_text = [" ".join(seed) for seed in seed_text]

  ##seed_text = [seed.lower() for seed in seed_text]

  x_seq_length = len(seedsWithReference[0][0].split(" ")) - 1 ####-1 oder nicht? TODO https://machinelearningmastery.com/how-to-develop-a-word-level-neural-language-model-in-keras/

  generated_texts = []

  for seed in seedsWithReference:
    generated_text = generate_text_seq(model, tokenizer, x_seq_length, seed[0], 40)
    generated_texts.append((seed[0], seed[1], generated_text))

  print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
  [print(text) for text in generated_texts]

  Path("target/").mkdir(parents=True, exist_ok=True)

  f = open(f"{targetFoldDir}/generated_texts.txt", "w")
  for text in generated_texts:
      # f.write(f"<SEED> {text[0]} <SEED_END> <REFERENCE> {text[1]} <REFERENCE_END> <GENERATED> {text[0]} {text[2]} <GENERATED_END>\n")
      f.write(f"<SEED> {text[0]} <SEED_END>\n")
  f.close()

