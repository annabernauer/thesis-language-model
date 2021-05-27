package com.mailytica.thesis.language.model.ngram.cosineSimilarity

import com.johnsnowlabs.nlp.{Annotation, LightPipeline}
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.nlp.util.io.ResourceHelper.spark.sqlContext
import com.mailytica.thesis.language.model.ngram.Timer.{consoleReporter, cosineSimilarityTimer, stopwatch}
import com.mailytica.thesis.language.model.ngram.cosineSimilarity.CosineSimilarityPipelines.{getPredictionStages, getPreprocessStages, getReferenceStages, getVectorizerStages}
import com.mailytica.thesis.language.model.ngram.pipelines.nGramSentences.ExecutableSentencePrediction.getClass
import com.mailytica.thesis.language.model.ngram.pipelines.nGramSentences.NGramSentencePrediction.getStages
import com.mailytica.thesis.language.model.util.Utility.{printToFile, srcName}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.{col, lit, size, udf}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

import java.io.{File, FileOutputStream, PrintStream}
import java.util.concurrent.TimeUnit
import scala.io.{Codec, Source}
import scala.io.StdIn.readLine
import scala.util.matching.Regex

object CosineExecutable {

  val REGEX_PUNCTUATION: Regex = "(\\.|\\!|\\?|\\,|\\:)$".r
  val n = 5

  val dirCrossfoldName = s"${srcName}_n_${n}"
  val specificDirectory = new File(s"target/crossFoldValues/$dirCrossfoldName")

  consoleReporter.start(1, TimeUnit.MINUTES)

  val spark: SparkSession = SparkSession
    .builder
    .config("spark.driver.maxResultSize", "5g")
    .config("spark.driver.memory", "12g")
    .config("spark.sql.codegen.wholeStage", "false") // deactivated as the compiled grows to big (exception)
    .master(s"local[3]") //threads = 6
    .getOrCreate()

  def main(args: Array[String]): Unit = {

    redirectConsoleLog()

    import spark.implicits._

    val nlpPipeline = new Pipeline()

    nlpPipeline.setStages(getPredictionStages(n))

    val path = s"src/main/resources/sentencePrediction/textsForTraining/bigData/$srcName.csv"

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
    val cosineCrossfoldAverages: Array[Double] = splitArray
      .take(2)
      .zipWithIndex
      .map { case (testData, fold) =>
        println("#################### index " + fold)

        val trainingData: DataFrame = splitArray
          .diff(Array(testData)) //remove testData
          .reduce(_ union _)
          .cache()

        val preprocessed =  prepocessData(testData) //get seed and reference and explode text into sentences
          .cache()
          .filter(!(col("seeds") <=> lit("")))

        writeTestAndTrainingsDataToFile(preprocessed, trainingData, fold)

        val pipelineModel: PipelineModel = nlpPipeline.fit(trainingData.toDF("seeds"))                                 //no preprocessing needed for trainingsData
        val predictionDf: DataFrame = pipelineModel.transform(preprocessed)


        val referenceProcessedDf: DataFrame = processReferenceData(predictionDf)                                      //remove new lines from reference, can't be removed before
                                                                                                                        //because they are needed for prediction
        val vectorizedData = vectorizeData(referenceProcessedDf).cache()
        vectorizedData.select("seeds","mergedPrediction", "referenceWithoutNewLines", "ngrams_reference", "ngrams_prediction", "cosine").show(20,false)
//        writeToFile(vectorizedData, fold)

        val cosineValues = vectorizedData
          .select("cosine")
          .as[Double]
          .collect()

        val crossfoldAverage = (cosineValues.sum) / cosineValues.length
        println(s"crossfoldAverage = $crossfoldAverage")

        printToFile(new File(s"${specificDirectory}/${dirCrossfoldName}_fold_${fold}/cosineValues")) { p =>
          cosineValues.foreach(p.println)
          p.println(s"crossfoldAverage = $crossfoldAverage")
        }

        crossfoldAverage
      }

    val totalCosineAvg = cosineCrossfoldAverages.sum / cosineCrossfoldAverages.length
    print(s"n = $n \ntotalCosineAvg = $totalCosineAvg")
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
    cosineSimilarity(vectorA, vectorB) //cosineSimilarity of each row
  }

  def cosineSimilarity(vectorA: Vector, vectorB: Vector) : Double = {

    stopwatch.reset()
    stopwatch.start()

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

    cosineSimilarityTimer.update(stopwatch.getTime, TimeUnit.MILLISECONDS)

    val div : Double = normASqrt * normBSqrt
    if( div == 0 )
      0
    else
      dotProduct / div
  }

  def writeTestAndTrainingsDataToFile(preprocessed: DataFrame, trainingData: DataFrame, fold: Int) = {

    val directoryFold = new File(s"${specificDirectory}/${dirCrossfoldName}_fold_${fold}")
    if (!directoryFold.exists) {
      directoryFold.mkdirs
    }

    val testDataFile = new File(directoryFold.getPath + s"/testData")
    val trainingDataFile = new File(directoryFold.getPath + s"/trainingData")

    preprocessed
      .select("referenceSentences", "seeds")
      .write
      .format("com.databricks.spark.csv")
      .option("header", "true")
      .save(testDataFile.getPath)

    println("INFO: Preprocessed data is saved")
//    trainingData
//      .toDF("seeds")
//      .write
//      .format("com.databricks.spark.csv")
//      .option("header", "true")
//      .save(trainingDataFile.getPath)
  }

  def writeToFile(data: DataFrame, fold: Int) = {
    val directoryFold = new File(s"${specificDirectory}/${dirCrossfoldName}_fold_${fold}")
    if (!directoryFold.exists) {
      directoryFold.mkdirs
    }

    val dataFile = new File(directoryFold.getPath + s"/evaluatedData")

    data
      .select("seeds","mergedPrediction", "referenceWithoutNewLines", "ngrams_reference", "ngrams_prediction","vectorizedCount_reference", "vectorizedCount_prediction" , "cosine")
      .write
      .format("com.databricks.spark.csv")
      .option("header", "true")
      .save(dataFile.getPath)

    println("INFO: Preprocessed data is saved")
  }

  def redirectConsoleLog() = {
    val console: PrintStream = System.out

    val logFile = new File(s"${specificDirectory}/log.txt")
    if (!specificDirectory.exists) {
      specificDirectory.mkdirs
    }

    val fos = new FileOutputStream(logFile)
    val ps = new PrintStream(fos)
//    System.setOut(ps)
    System.setOut(console)
  }
}
