package com.mailytica.thesis.language.model.ngram.pipelines.textSplittingDemo

;

import org.apache.spark.sql.{DataFrame, SparkSession}
import com.johnsnowlabs.nlp.DocumentAssembler
import com.johnsnowlabs.nlp.annotator.{Lemmatizer, Normalizer, Stemmer, StopWordsCleaner, Tokenizer}
import com.johnsnowlabs.nlp.annotators.DateMatcher
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import org.apache.spark.ml.{Pipeline, PipelineModel}

object TextSplitting extends App {

  val sparkSession = SparkSession
    .builder
    .config("spark.sql.codegen.wholeStage", "false") // deactivated as the compiled grows to big (exception)
    .master(s"local[1]")
    .getOrCreate()

  import sparkSession.implicits._

  val data = Seq("Test. data", "testing Data2 1/12/20 last wednesday the day before next thursday").toDF("text")
  data.show()

  val documentAssembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

  val sentenceDetector = new SentenceDetector()
    .setInputCols("document")
    .setOutputCol("sentences")

  val tokenizer = new Tokenizer()
    .setInputCols("sentences")
    .setOutputCol("token")

  val normalizer = new Normalizer()
    .setInputCols("token")
    .setOutputCol("normalized")
    .setLowercase(true)

  val stopwordsCleaner = new StopWordsCleaner()
    .setInputCols("normalized")
    .setOutputCol("removed_stopwords")
    .setCaseSensitive(false)

  val stemmer = new Stemmer()
    .setInputCols("removed_stopwords")
    .setOutputCol("stem")

  val lemmatizer = new Lemmatizer()
    .setInputCols("removed_stopwords")
    .setOutputCol("lemma")
    .setDictionary(
      "n-gram-language-model\\src\\main\\resources\\textSplittingDemo\\AntBNC_lemmas_ver_001.txt",
      "\t", "->")

  val dateMatcher = new DateMatcher()
    .setInputCols("sentences")
    .setOutputCol("dateMatched")
    .setFormat("dd.MM.yyyy")
    .setAnchorDateDay(11)
    .setAnchorDateMonth(1)
    .setAnchorDateYear(2021)


  val pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, normalizer,
    stopwordsCleaner, stemmer, lemmatizer, dateMatcher))

  // train
  val pipelineModel: PipelineModel = pipeline.fit(data)

  // predict
  val nlpData: DataFrame = pipelineModel.transform(data)

  nlpData.show(false)

//  nlpData.select("sentences").show(false)
  nlpData.select("dateMatched").show(false)

}
