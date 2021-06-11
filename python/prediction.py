
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import tensorflow as tf
from pickle import load
from pathlib import Path
from pyspark.sql import *
import glob
import logging

def generate_text_seq(model, tokenizer, text_seq_length, seed_text, n_words):
  text = []

  # for _ in range(n_words):
  i = 0
  while i < n_words:
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
    if predicted_word == "<SENTENCE_END>"or predicted_word == "<sentence_end>":
      break
    i += 1
  return ' '.join(text)




logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

spark = SparkSession \
  .builder \
  .appName("Python Spark SQL basic example") \
  .config("spark.some.config.option", "some-value") \
  .getOrCreate()

n = 5
epochs = 25  #30 old value
embeddings = 100
src_name = "messages"
foldCount = 10

logging.info(f"n = {n}, epochs = {epochs}, embeddings = {embeddings}, src_name = {src_name}, foldCount = {foldCount}")

srcName = f"{src_name}_n_{n}"

for fold in range(foldCount):

  logging.info(f"+++++++++++++++++++++++++++++ fold = {fold} +++++++++++++++++++++++++++++")

  foldDir = f"{srcName}_fold_{fold}"
  targetFoldDir = f"target/{srcName}_emb_{embeddings}_epo_{epochs}/{foldDir}"

  # load the model
  model = load_model(f'{targetFoldDir}/model.h5')

  # load the tokenizer
  tokenizer = load(open(f'{targetFoldDir}/tokenizer.pkl', 'rb'))

  df = spark.read.csv(glob.glob(f'resources/{srcName}/{foldDir}/testData/*.csv'), header="true", inferSchema="true")

  # seeds = ("Mögliche Änderungswünsche nehmen wir sehr",
  # "Sollten Sie weitere Fragen haben",
  # "Alternativ würde ich mich jederzeit")

  # result_lst = df.rdd.map(lambda row: row.getString(0)).collect()
  # [print(x) for x in result_lst]

  seeds = [str(row.seeds) for row in df.select("seeds").collect()]
  reference = [str(row.referenceSentences) for row in df.select("referenceSentences").collect()]

  seedsWithReference = list(zip(seeds, reference))
  # [print(x) for x in seedsWithReference]

  # ############
  # seedsTest = (
  # "<SENTENCE_START> vielen Dank für",
  # "<SENTENCE_START> Sehr geehrte Frau",
  # "<SENTENCE_START> hiermit bedanke ich")
  #
  # referenceTest = (
  #   "<SENTENCE_START> vielen Dank für Ihre Nachfrage hinsichtlich des Lieferdatums Ihrer Bestellung. <SENTENCE_END>",
  #   "<SENTENCE_START> Sehr geehrte Frau Hilgers, <SENTENCE_END>",
  #   "<SENTENCE_START> hiermit bedanke ich mich für Ihre Bestellung. <SENTENCE_END>"
  # )
  # seedsWithReference = list(zip(seedsTest, referenceTest))
  # # [print(x) for x in seedsWithReference]
  # ###########

  logging.info(f"seedsWIthReference: {seedsWithReference[0][0]}; {seedsWithReference[0][1]}")


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

  count = 0
  for seed in seedsWithReference:
    generated_text = generate_text_seq(model, tokenizer, x_seq_length, seed[0], 40)
    generated_texts.append((seed[0], seed[1], generated_text))
    if (count % 50) == 0:
      logging.info(f"count: {count}; {seed[0]}, {seed[1]}, {generated_text}")
    count = count + 1

  # print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
  # [print(text) for text in generated_texts]

  Path("target/").mkdir(parents=True, exist_ok=True)

  logging.info(f"<SEED> {generated_texts[0][0]} <SEED_END> <REFERENCE> {generated_texts[0][1]} <REFERENCE_END> <GENERATED> {generated_texts[0][0]} {generated_texts[0][2]} <GENERATED_END>")
  f = open(f"{targetFoldDir}/generated_texts.txt", "w")
  for text in generated_texts:
      f.write(f"<SEED> {text[0]} <SEED_END> <REFERENCE> {text[1]} <REFERENCE_END> <GENERATED> {text[0]} {text[2]} <GENERATED_END>\n")
      # f.write(f"<SEED> {text[0]} <SEED_END>\n")
  f.close()

  logging.info(f"fold {fold} finished, file is saved")

