package com.mailytica.thesis.language.model.ngram.cosineSimilarity

import com.johnsnowlabs.nlp.{Annotation, LightPipeline}
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.nlp.util.io.ResourceHelper.spark.sqlContext
import com.mailytica.thesis.language.model.ngram.cosineSimilarity.CosineSimilarityPipelines.{getPredictionStages, getPreprocessStages, getReferenceStages}
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

  val spark = SparkSession
    .builder
    .config("spark.driver.maxResultSize", "5g")
    .config("spark.driver.memory", "12g")
    .config("spark.sql.codegen.wholeStage", "false") // deactivated as the compiled grows to big (exception)
    .master(s"local[3]")
    .getOrCreate()

  import spark.implicits._


  def main(args: Array[String]): Unit = {

    import spark.implicits._

    val nlpPipeline = new Pipeline()

    nlpPipeline.setStages(getPredictionStages(n))

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

        val (context: Seq[String], reference: Seq[String]) = prepocessData(testData)
        val annotated: DataFrame = pipelineModel.transform(context.zip(reference).toDF("text", "reference"))

        val predictions: Array[String] = getCol(annotated, "mergedPrediction")
        //          val referencesProcessed : Array[String] = getCol(annotated, "referenceWithoutNewLines")

        val referenceProcessed = processReferenceData(reference)


        val contextReferencePredictions: Seq[ContextReferencePrediction] = {
          context
            .zip(referenceProcessed)
            .zip(predictions)
            .map(contextReferencePrediction =>
              ContextReferencePrediction(contextReferencePrediction._1._1, contextReferencePrediction._1._2, contextReferencePrediction._2))
        }

        contextReferencePredictions.foreach(println)

      }
  }

  def getCol(df: DataFrame, colName: String): Array[String] = {
    df
      .select(colName)
      .cache()
      .as[Array[Annotation]]
      .collect()
      .flatten
      //            .filter(annotation => annotation.result != "empty")
      .map(annotation => annotation.result)
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

  def prepocessData(data: Dataset[Row]) = {
    val nlpPipelinePreprocess = new Pipeline()

    nlpPipelinePreprocess.setStages(getPreprocessStages(n))
    val pipelineModel: PipelineModel = nlpPipelinePreprocess.fit(data.toDF("text"))

    val processed: DataFrame = pipelineModel.transform(data.toDF("text"))

    val processedSeeds: DataFrame = processed.select("seeds")
    val processedSentences: DataFrame = processed.select("explodedDocument")

    val seedAnnotations: Seq[String] = processedSeeds
      .as[Array[Annotation]]
      .collect()
      .flatten
      //      .filter(annotation => annotation.result != "empty")
      .toSeq
      .map(annotation => annotation.result)

    val sentenceAnnotation: Seq[String] = processedSentences
      .as[Array[Annotation]]
      .collect()
      .flatten
      //      .filter(annotation => annotation.result != "empty")
      .toSeq
      .map(annotation => annotation.result)

    (seedAnnotations, sentenceAnnotation)
  }

  def processReferenceData(references: Seq[String]) = {
    val nlpPipelineReference = new Pipeline()

    nlpPipelineReference.setStages(getReferenceStages())
    val pipelineModel: PipelineModel = nlpPipelineReference.fit(references.toDF("reference"))

    val processed: DataFrame = pipelineModel.transform(references.toDF("reference"))

    val processedReference: DataFrame = processed.select("referenceWithoutNewLines")

    processedReference
      .as[Array[Annotation]]
      .collect()
      .flatten
      //      .filter(annotation => annotation.result != "empty")
      .toSeq
      .map(annotation => annotation.result)
  }
}
