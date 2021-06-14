package com.mailytica.thesis.language.model.ngram.cosineSimilarity

import com.codahale.metrics.MetricRegistry
import com.johnsnowlabs.nlp.util.io.ResourceHelper.spark.sqlContext
import com.mailytica.thesis.language.model.ngram.Timer.{slf4jReporter}
import com.mailytica.thesis.language.model.ngram.cosineSimilarity.pipelines.CosineSimilarityPipelines.{getPredictionStages, getPreprocessStages, getReferenceStages}
import com.mailytica.thesis.language.model.util.Utility.{printToFile, srcName}
import org.apache.commons.io.FileUtils
import org.apache.commons.lang.time.StopWatch
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.functions.{col, lit}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.slf4j.LoggerFactory

import java.io.{File, FileOutputStream, PrintStream}
import java.util.concurrent.TimeUnit
import scala.collection.parallel.mutable.ParArray
import scala.io.StdIn.readLine
import scala.util.matching.Regex
import scala.collection.parallel._

object CosineExecutable {

  val logger = LoggerFactory.getLogger(this.getClass)

  val REGEX_PUNCTUATION: Regex = "(\\.|\\!|\\?|\\,|\\:)$".r
  val n = 3

  val dirCrossfoldName = s"${srcName}_n_${n}"
  val specificDirectory = new File(s"target/crossFoldValues/$dirCrossfoldName")

  if (specificDirectory.exists()) {
    logger.info(s"Directory ${dirCrossfoldName} already exists. Do you want to overwrite it? [y|n]")
    val input: String = readLine()
    input.charAt(0) match {
      case 'y' => FileUtils.deleteQuietly(specificDirectory)
      case _ =>
    }
  }

  slf4jReporter.start(5, TimeUnit.MINUTES)

  val spark: SparkSession = SparkSession
    .builder
    .config("spark.driver.maxResultSize", "5g")
    .config("spark.driver.memory", "12g")
    .config("spark.sql.codegen.wholeStage", "false") // deactivated as the compiled grows to big (exception)
    .master(s"local[10]") //threads = 6
    .getOrCreate()

  def main(args: Array[String]): Unit = {

    logger.info(s"n = $n")
    logger.info(s"srcFile = $srcName")

    redirectConsoleLog()

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

    val numOfThreads = 2
    val parColl = splitArray
      //      .take(4)
      .zipWithIndex
      .par

    parColl.tasksupport = new ForkJoinTaskSupport(new scala.concurrent.forkjoin.ForkJoinPool(numOfThreads))

    val cosineCrossfoldAverages: ParArray[Double] =
      parColl
      .map { case (testData, fold) =>
        logger.info("#################### index " + fold)

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

        logger.debug(s"fold = $fold crossfoldAverage = $crossfoldAverage")

        printToFile(new File(s"${specificDirectory}/${dirCrossfoldName}_fold_${fold}/cosineValues")) { p =>
          p.println(s"fold = $fold")
          cosineValues.foreach(p.println)
          p.println(s"fold = $fold crossfoldAverage = $crossfoldAverage")
        }

        crossfoldAverage
      }

    val totalCosineAvg = cosineCrossfoldAverages.sum / cosineCrossfoldAverages.length
    logger.info("ALL CROSSFOLD AVERAGES")
    cosineCrossfoldAverages.foreach(foldAverage => logger.info(s"foldAvg = $foldAverage"))
    logger.info(s"\nn = $n \ntotalCosineAvg = $totalCosineAvg")
    logger.info(s"srcFile = $srcName")
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

    logger.info(s"INFO: Preprocessed data is saved, fold = $fold")
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

    logger.info(s"INFO: Data is saved, fold = $fold")
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
