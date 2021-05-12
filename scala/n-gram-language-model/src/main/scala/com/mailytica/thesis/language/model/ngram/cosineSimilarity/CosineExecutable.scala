package com.mailytica.thesis.language.model.ngram.cosineSimilarity

import com.johnsnowlabs.nlp.{Annotation, LightPipeline}
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.nlp.util.io.ResourceHelper.spark.sqlContext
import com.mailytica.thesis.language.model.ngram.cosineSimilarity.PreprocessTestDataPipeline.getPreprocessStages
import com.mailytica.thesis.language.model.ngram.pipelines.nGramSentences.ExecutableSentencePrediction.getClass
import com.mailytica.thesis.language.model.ngram.pipelines.nGramSentences.NGramSentencePrediction.getStages
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

import scala.io.{Codec, Source}
import scala.io.StdIn.readLine
import scala.util.matching.Regex

object CosineExecutable {
  val REGEX_PUNCTUATION: Regex = "(\\.|\\!|\\?|\\,|\\:)$".r
  val n = 5

  def main(args: Array[String]): Unit = {

    val spark = SparkSession
      .builder
      .config("spark.driver.maxResultSize", "5g")
      .config("spark.driver.memory", "12g")
      .config("spark.sql.codegen.wholeStage", "false") // deactivated as the compiled grows to big (exception)
      .master(s"local[3]")
      .getOrCreate()

    import spark.implicits._

    val nlpPipeline = new Pipeline()

    nlpPipeline.setStages(getStages(n))

    val path = "src/main/resources/sentencePrediction/textsForTraining/bigData/messagesSmall.csv"

    val df: DataFrame = sqlContext.read.format("com.databricks.spark.csv")
      .option("header", "true")
      .option("quote", "\"")
      .option("escape", "\\")
      .option("multiLine", value = true)
      .load(path)

    val fraction = 1.0 / 10.toDouble
    val fractionPerSplit = Array.fill(10)(fraction)
    val splitArray: Array[Dataset[Row]] = df.randomSplit(fractionPerSplit)

//    val allCrossFoldValues: Array[MetadataTypes] =
      splitArray
        .take(1)
        .foreach { testData =>

          val trainingData: DataFrame = splitArray
            .diff(Array(testData)) //remove testData
            .reduce(_ union _)

          val pipelineModel: PipelineModel = nlpPipeline.fit(trainingData.toDF("text"))

          val preprocessedTestData: Unit = prepocessData(testData, spark)
//          println("+++++++++++++++++++++++++")
//          preprocessedTestData.show(false)
//          println("###########################")
          val annotated: DataFrame = pipelineModel.transform(trainingData.toDF("text"))

          val processed = annotated
            .select("sentencePrediction")
            .cache()

          //      processed.show(100, false)

          val annotationsPerDocuments: Array[Annotation] = processed
            .as[Array[Annotation]]
            .collect()
            .flatten
            .filter(annotation => annotation.result != "empty")

        }
  }

  def getResourceText(path: String) = {
    Seq.range(0, 9).map {
      x => {
        resource
          .managed(getClass.getResourceAsStream(s"$path/00$x.txt"))
          .acquireAndGet(inputStream => {

            Source
              .fromInputStream(inputStream)(Codec.UTF8)
              .mkString + " <SENTENCE_END>"
          })
      }
    }
  }

  def prepocessData(data: Dataset[Row], spark: SparkSession) = {
    val nlpPipelinePreprocess = new Pipeline()

    import spark.implicits._

    nlpPipelinePreprocess.setStages(getPreprocessStages(n))
    val pipelineModel: PipelineModel = nlpPipelinePreprocess.fit(data.toDF("text"))

    val processed = pipelineModel.transform(data.toDF("text")).select("seeds")

    val annotationsPerDocuments: Array[Annotation] = processed
      .as[Array[Annotation]]
      .collect()
      .flatten
      .filter(annotation => annotation.result != "empty")

    annotationsPerDocuments.foreach(println)
    annotationsPerDocuments.map(annotation => annotation.result)
  }
}
