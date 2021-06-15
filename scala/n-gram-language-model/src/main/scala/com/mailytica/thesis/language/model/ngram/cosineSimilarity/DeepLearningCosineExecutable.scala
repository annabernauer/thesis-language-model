package com.mailytica.thesis.language.model.ngram.cosineSimilarity

import com.mailytica.thesis.language.model.ngram.cosineSimilarity.CosineExecutable.n
import com.mailytica.thesis.language.model.util.Utility.{printToFile, srcName}
import org.apache.spark.sql.SparkSession
import org.slf4j.LoggerFactory

import java.io.{BufferedReader, File, FileInputStream, InputStreamReader}
import scala.collection.parallel.ForkJoinTaskSupport
import scala.collection.parallel.mutable.ParArray

object DeepLearningCosineExecutable {

  val logger = LoggerFactory.getLogger(this.getClass)

  val sparkSession: SparkSession = SparkSession
    .builder
    .config("spark.driver.maxResultSize", "5g")
    .config("spark.driver.memory", "12g")
    .config("spark.sql.codegen.wholeStage", "false") // deactivated as the compiled grows to big (exception)
    .master(s"local[10]") //threads = 6
    .getOrCreate()

  val SEED_REGEX = "(?<=<SEED>)(.*?)(?=<SEED_END>)".r()
  val REFERENCE_REGEX = "(?<=<REFERENCE>)(.*?)(?=<REFERENCE_END>)".r()
  val GENERATED_REGEX = "(?<=<GENERATED>)(.*?)(?=<GENERATED_END>)".r()

  val n = 6
  val emb = 100
  val epo = 25
  val srcName = "messages"

  val dirCrossfoldName = s"${srcName}_n_${n}"
  val specificTargetDirectory = new File(s"target/crossFoldValues/$dirCrossfoldName")

  def main(args: Array[String]): Unit = {

    logger.info("in DeepLearningCosineExecutable")
    logger.info(s"n = $n, emb = $emb, epo = $epo")
    logger.info(s"srcFile = $srcName")

    val numOfThreads = 2
    val parArray = Array.range(0, 10).par
    parArray.tasksupport = new ForkJoinTaskSupport(new scala.concurrent.forkjoin.ForkJoinPool(numOfThreads))

    val cosineCrossfoldAverages: ParArray[Double] = parArray.map {
      fold =>
        logger.info("#################### index " + fold)
        val reader = new BufferedReader(new InputStreamReader(new FileInputStream(s"src/main/resources/sentencePrediction/deepLearningGeneratedTexts/${srcName}_n_${n}_emb_${emb}_epo_${epo}/${srcName}_n_${n}_fold_${fold}/generated_texts.txt"), "Cp1252"))
        val lines: Seq[String] = Stream.continually(reader.readLine()).takeWhile(_ != null)

        val seperatedLines: Seq[SentenceResult] = lines.map {
          line =>
            val seed = SEED_REGEX.findFirstIn(line).getOrElse("")
            val reference = REFERENCE_REGEX.findFirstIn(line).getOrElse("")
            val generated = GENERATED_REGEX.findFirstIn(line).getOrElse("")
            SentenceResult(seed, reference, generated)
        }

        import sparkSession.implicits._

        val data = seperatedLines.map(sentenceResult => (sentenceResult.seed, sentenceResult.reference, sentenceResult.generated)).toDF("seeds", "reference", "generated")

        val vectorizedData = CosineSimilarity.vectorizeData(data, "generated", "reference", needsDocAssembl = true)

        val (cosineValues, crossfoldAverage) = CosineSimilarity.calculateCosineValues(vectorizedData, "generated", "reference", sparkSession)

        logger.debug(s"fold = $fold crossfoldAverage = $crossfoldAverage")

        //    printToFile(new File(s"${specificDirectory}/${dirCrossfoldName}_fold_${fold}/cosineValues")) { p =>

        val dir = new File(s"${specificTargetDirectory}/${dirCrossfoldName}_fold_$fold")
        if (!dir.exists) {
          dir.mkdirs
        }
        printToFile(new File(dir.getPath + "/cosineValues_deep_learning")) { p =>
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

}
