package com.mailytica.thesis.language.model.ngram.cosineSimilarity

import com.johnsnowlabs.nlp.{Annotation, LightPipeline}
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.nlp.util.io.ResourceHelper.spark.sqlContext
import com.mailytica.thesis.language.model.ngram.Timer.{consoleReporter, cosineDotProduct, cosineNormASqurt, cosineNormBSqurt, cosineSimilarityTimer, stopwatch}
import com.mailytica.thesis.language.model.ngram.cosineSimilarity.CosineSimilarity.calculateCosineValues
import com.mailytica.thesis.language.model.ngram.cosineSimilarity.pipelines.CosineSimilarityPipelines.{getPredictionStages, getPreprocessStages, getReferenceStages, getVectorizerStages}
import com.mailytica.thesis.language.model.ngram.pipelines.nGramSentences.ExecutableSentencePrediction.getClass
import com.mailytica.thesis.language.model.ngram.pipelines.nGramSentences.NGramSentencePrediction.getStages
import com.mailytica.thesis.language.model.util.Utility.{printToFile, srcName}
import org.apache.commons.io.FileUtils
import org.apache.commons.lang.time.StopWatch
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.{col, lit, size, udf}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

import java.io.{File, FileOutputStream, PrintStream}
import java.util.concurrent.TimeUnit
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.duration.Duration
import scala.concurrent.{Await, Future}
import scala.io.{Codec, Source}
import scala.io.StdIn.readLine
import scala.util.matching.Regex

object CosineExecutable {

  val REGEX_PUNCTUATION: Regex = "(\\.|\\!|\\?|\\,|\\:)$".r
  val n = 5

  val dirCrossfoldName = s"${srcName}_n_${n}"
  val specificDirectory = new File(s"target/crossFoldValues/$dirCrossfoldName")

  if (specificDirectory.exists()) {
    println(s"Directory ${dirCrossfoldName} already exists. Do you want to overwrite it? [y|n]")
    val input: String = readLine()
    input.charAt(0) match {
      case 'y' => FileUtils.deleteQuietly(specificDirectory)
      case _ =>
    }
  }

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
//      .take(4)
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
        val vectorizedData = CosineSimilarity.vectorizeData(referenceProcessedDf, "mergedPrediction", "referenceWithoutNewLines", needsDocAssembl = false).cache()

        val (cosineValues, crossfoldAverage) = CosineSimilarity.calculateCosineValues(vectorizedData, "mergedPrediction", "referenceWithoutNewLines", spark)

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
    System.setOut(ps)
//    System.setOut(console)
  }
}
