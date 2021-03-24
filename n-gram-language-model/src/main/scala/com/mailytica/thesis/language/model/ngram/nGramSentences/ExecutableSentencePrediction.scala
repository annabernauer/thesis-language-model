package com.mailytica.thesis.language.model.ngram.nGramSentences

import com.johnsnowlabs.nlp.{Annotation, LightPipeline}
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.mailytica.thesis.language.model.ngram.nGramSentences.NGramSentencePrediction.getStages
import org.apache.spark.ml.{Pipeline, PipelineModel}

import scala.io.{Codec, Source}
import scala.io.StdIn.readLine
import scala.util.matching.Regex

object ExecutableSentencePrediction {

  val REGEX_PUNCTUATION: Regex = "(\\.|\\!|\\?|\\,|\\:)$".r

  def main(args: Array[String]): Unit = {

    ResourceHelper.spark

    val nlpPipeline = new Pipeline()

    nlpPipeline.setStages(getStages(4))

    val texts : Seq[String] = getResourceText("/sentencePrediction/textsForTraining/productionRelease")

    val texts2 : Seq[String] = getResourceText("/sentencePrediction/textsForTraining/shippingNotification")

    while (true) {
      val input = readLine("Please type your text\n")

      import ResourceHelper.spark.implicits._

      val pipelineModel: PipelineModel = nlpPipeline.fit((texts ++ texts2).toDF("text"))

      val annotated: Map[String, Seq[Annotation]] = new LightPipeline(pipelineModel).fullAnnotate(input)

      println("\nSentence prediction: ")

      annotated("sentencePrediction").foreach(x => {
        REGEX_PUNCTUATION.findFirstMatchIn(x.result) match {
          case None => print(s" ${x.result}")
          case Some(_) => print(s"${x.result}")
        }
//        print(s"${x.result} ")
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
