package com.mailytica.thesis.language.model.ngram.pipelines.nGramSentences

import com.johnsnowlabs.nlp.{Annotation, LightPipeline}
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.nlp.util.io.ResourceHelper.spark.sqlContext
import com.mailytica.thesis.language.model.ngram.pipelines.nGramSentences.NGramSentencePrediction.getStages
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.DataFrame

import scala.io.{Codec, Source}
import scala.io.StdIn.readLine
import scala.util.matching.Regex

object ExecutableSentencePrediction {

  val REGEX_PUNCTUATION: Regex = "(\\.|\\!|\\?|\\,|\\:)$".r

  def main(args: Array[String]): Unit = {

    ResourceHelper.spark

    val nlpPipeline = new Pipeline()

    nlpPipeline.setStages(getStages(7))

    val texts : Seq[String] = getResourceText("/sentencePrediction/textsForTraining/productionRelease")

    val texts2 : Seq[String] = getResourceText("/sentencePrediction/textsForTraining/shippingNotification")

    val path = "src/main/resources/sentencePrediction/textsForTraining/bigData/messagesSmall.csv"

    val df: DataFrame = sqlContext.read.format("com.databricks.spark.csv")
      .option("header", "true")
      .option("quote", "\"")
      .option("escape", "\\")
      .option("multiLine", value = true)
      .load(path)

    df.show()

    val pipelineModel: PipelineModel = nlpPipeline.fit(df.toDF("text"))

    while (true) {
      val input = readLine("Please type your text\n")

      val annotated: Map[String, Seq[Annotation]] = new LightPipeline(pipelineModel).fullAnnotate(input)    //colName and Seq of Annotations

      println("\nSentence prediction: ")

      annotated("sentencePrediction").foreach(token => {
        REGEX_PUNCTUATION.findFirstMatchIn(token.result) match {
          case None => print(s" ${token.result}")
          case Some(_) => print(s"${token.result}")     //punctuation without whitespace
        }
//        print(s"${token.result} ")
      })

      print("\n\n")

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

}
