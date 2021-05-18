package com.mailytica.thesis.language.model.ngram.cosineSimilarity

import com.johnsnowlabs.nlp.{Annotation, LightPipeline}
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.nlp.util.io.ResourceHelper.spark.sqlContext
import com.mailytica.thesis.language.model.ngram.cosineSimilarity.CosineSimilarityPipelines.{getPredictionStages, getPreprocessStages, getReferenceStages, getVectorizerStages}
import com.mailytica.thesis.language.model.ngram.pipelines.nGramSentences.ExecutableSentencePrediction.getClass
import com.mailytica.thesis.language.model.ngram.pipelines.nGramSentences.NGramSentencePrediction.getStages
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.{col, udf}
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

        val preprocessed =  prepocessData(testData)                                                                      //get seed and reference and explode text into sentences

        val pipelineModel: PipelineModel = nlpPipeline.fit(trainingData.toDF("seeds"))                                 //no preprocessing needed for trainingsData
        val predictionDf: DataFrame = pipelineModel.transform(preprocessed)


        val referenceProcessedDf: DataFrame = processReferenceData(predictionDf)                                      //remove new lines from reference, can't be removed before
                                                                                                                        //because they are needed for prediction
        val vectorizedData = vectorizeData(referenceProcessedDf)
//        vectorizedData.select("seeds","mergedPrediction", "referenceWithoutNewLines", "cosine").show(100,false)
        vectorizedData.show(100)

      }
  }


  def prepocessData(data: Dataset[Row]) = {
    val preprocessPipeline = new Pipeline()

    preprocessPipeline.setStages(getPreprocessStages(n))
    val pipelineModel: PipelineModel = preprocessPipeline.fit(data.toDF("data"))

    val processed: DataFrame = pipelineModel.transform(data.toDF("data"))

    processed
  }


  def processReferenceData(df: DataFrame) = {
    val nlpPipelineReference = new Pipeline()

    nlpPipelineReference.setStages(getReferenceStages())
    val pipelineModel: PipelineModel = nlpPipelineReference.fit(df)

    val processed: DataFrame = pipelineModel.transform(df)

    processed
  }

  def vectorizeData(df: DataFrame) = {

    val vectorizePipeline = new Pipeline()
    vectorizePipeline.setStages(
      getVectorizerStages("mergedPrediction", "prediction") ++
        getVectorizerStages("referenceWithoutNewLines", "reference"))

    val pipelineModel: PipelineModel = vectorizePipeline.fit(df)
    val annotatedHypothesis: DataFrame = pipelineModel.transform(df)

    val withCosineColumn: DataFrame = annotatedHypothesis.withColumn("cosine", cosineSimilarityUdf(col("vectorizedCount_prediction"), col("vectorizedCount_reference")))
    withCosineColumn
  }

  val cosineSimilarityUdf : UserDefinedFunction = udf{ (vectorA : Vector, vectorB: Vector) =>
    cosineSimilarity(vectorA, vectorB)
  }

  def cosineSimilarity(vectorA: Vector, vectorB: Vector) : Double = {

    val vectorArrayA = vectorA.toArray
    val vectorArrayB = vectorB.toArray

    val normASqrt : Double = Math.sqrt(vectorArrayA.map{ value =>
      Math.pow(value , 2)
    }.sum)

    val normBSqrt : Double = Math.sqrt(vectorArrayB.map{ value =>
      Math.pow(value , 2)
    }.sum)

    val dotProduct : Double =  vectorArrayA
      .zip(vectorArrayB)
      .map{case (x,y) => x*y }
      .sum

    val div : Double = normASqrt * normBSqrt
    if( div == 0 )
      0
    else
      dotProduct / div
  }


}
