package com.mailytica.thesis.language.model.ngram.cosineSimilarity

import com.mailytica.thesis.language.model.util.Utility.{printToFile, srcName}
import org.apache.spark.sql.{DataFrame, SparkSession}

import java.nio.charset.{Charset, CodingErrorAction}
import scala.collection.immutable
import scala.io.{Codec, Source}
import java.io.{BufferedReader, File, FileInputStream, InputStreamReader}

object DeepLearningCosineExecutable {

  val sparkSession: SparkSession = SparkSession
    .builder
    .config("spark.driver.maxResultSize", "5g")
    .config("spark.driver.memory", "12g")
    .config("spark.sql.codegen.wholeStage", "false") // deactivated as the compiled grows to big (exception)
    .master(s"local[3]") //threads = 6
    .getOrCreate()

  val SEED_REGEX = "(?<=<SEED>)(.*?)(?=<SEED_END>)".r()
  val REFERENCE_REGEX = "(?<=<REFERENCE>)(.*?)(?=<REFERENCE_END>)".r()
  val GENERATED_REGEX = "(?<=<GENERATED>)(.*?)(?=<GENERATED_END>)".r()

  val n = 5

  val dirCrossfoldName = s"${srcName}_n_${n}"
  val specificDirectory = new File(s"target/crossFoldValues/$dirCrossfoldName")


  def main(args: Array[String]): Unit = {
    val reader = new BufferedReader(new InputStreamReader(new FileInputStream("src/main/resources/sentencePrediction/deepLearningGeneratedTexts/messagesSmall_n_5_emb_100_epo_20/messagesSmall_n_5_fold_0/generated_texts.txt"), "Cp1252"))
    val lines: Seq[String] = Stream.continually(reader.readLine()).takeWhile(_ != null)
    lines.foreach(println)

    val seperatedLines: Seq[SentenceResult] = lines.map {
      line =>
        val seed = SEED_REGEX.findFirstIn(line).getOrElse("")
        val reference = REFERENCE_REGEX.findFirstIn(line).getOrElse("")
        val generated = GENERATED_REGEX.findFirstIn(line).getOrElse("")
        SentenceResult(seed, reference, generated)
    }

    seperatedLines.foreach(println)

    import sparkSession.implicits._

    val data = seperatedLines.map(sentenceResult => (sentenceResult.seed, sentenceResult.reference, sentenceResult.generated)).toDF("seeds", "reference", "generated")

    val vectorizedData = CosineSimilarity.vectorizeData(data, "generated", "reference", needsDocAssembl = true)

    val (cosineValues, crossfoldAverage) = CosineSimilarity.calculateCosineValues(vectorizedData, "generated", "reference", sparkSession)

    println(s"crossfoldAverage = $crossfoldAverage")

    //    printToFile(new File(s"${specificDirectory}/${dirCrossfoldName}_fold_${fold}/cosineValues")) { p =>

    val dir = new File(s"${specificDirectory}/${dirCrossfoldName}_fold_0")
    if (!dir.exists) {
      dir.mkdirs
    }
    printToFile(new File(dir.getPath + "/cosineValues_deep_learning")) { p =>
      cosineValues.foreach(p.println)
      p.println(s"crossfoldAverage = $crossfoldAverage")
    }

    crossfoldAverage
  }

}
