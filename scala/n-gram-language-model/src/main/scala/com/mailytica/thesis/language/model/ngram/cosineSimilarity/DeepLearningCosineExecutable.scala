package com.mailytica.thesis.language.model.ngram.cosineSimilarity

import com.mailytica.thesis.language.model.ngram.cosineSimilarity.CosineExecutable.n
import com.mailytica.thesis.language.model.util.Utility.{printToFile}
import org.apache.spark.sql.SparkSession

import java.io.{BufferedReader, File, FileInputStream, InputStreamReader}

object DeepLearningCosineExecutable {

  val sparkSession: SparkSession = SparkSession
    .builder
    .config("spark.driver.maxResultSize", "5g")
    .config("spark.driver.memory", "12g")
    .config("spark.sql.codegen.wholeStage", "false") // deactivated as the compiled grows to big (exception)
    .master(s"local[6]") //threads = 6
    .getOrCreate()

  val SEED_REGEX = "(?<=<SEED>)(.*?)(?=<SEED_END>)".r()
  val REFERENCE_REGEX = "(?<=<REFERENCE>)(.*?)(?=<REFERENCE_END>)".r()
  val GENERATED_REGEX = "(?<=<GENERATED>)(.*?)(?=<GENERATED_END>)".r()

  val n = 5
  val emb = 100
  val epo = 20
  val srcName = "messagesSmall"

  val dirCrossfoldName = s"${srcName}_n_${n}"
  val specificTargetDirectory = new File(s"target/crossFoldValues/$dirCrossfoldName")

  def main(args: Array[String]): Unit = {

    val cosineCrossfoldAverages: Array[Double] = Array.range(0, 9).map {
      fold =>

        val reader = new BufferedReader(new InputStreamReader(new FileInputStream(s"src/main/resources/sentencePrediction/deepLearningGeneratedTexts/${srcName}_n_${n}_emb_${emb}_epo_${epo}/${srcName}_n_${n}_fold_${fold}/generated_texts.txt"), "Cp1252"))
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

        val dir = new File(s"${specificTargetDirectory}/${dirCrossfoldName}_fold_$fold")
        if (!dir.exists) {
          dir.mkdirs
        }
        printToFile(new File(dir.getPath + "/cosineValues_deep_learning")) { p =>
          cosineValues.foreach(p.println)
          p.println(s"crossfoldAverage = $crossfoldAverage")
        }

        crossfoldAverage
    }
    val totalCosineAvg = cosineCrossfoldAverages.sum / cosineCrossfoldAverages.length
    print(s"n = $n \ntotalCosineAvg = $totalCosineAvg")
  }

}
