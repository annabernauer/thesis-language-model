package com.mailytica.thesis.language.model.ngram.nGramSentences

import com.johnsnowlabs.nlp.{Annotation, LightPipeline}
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.mailytica.thesis.language.model.ngram.nGramSentences.NGramSentencePrediction.getStages
import org.apache.spark.ml.{Pipeline, PipelineModel}

import scala.io.StdIn.readLine

object ExecutableSentencePrediction {

  def main(args: Array[String]): Unit = {

    ResourceHelper.spark

    import ResourceHelper.spark.implicits._

    val nlpPipeline = new Pipeline()

    nlpPipeline.setStages(getStages())

    val texts : Seq[String] = Seq.range(0, 9).map {
      x => {
        val source = scala.io.Source.fromFile(s"n-gram-language-model\\src\\main\\resources\\sentencePrediction\\textsForTraining\\productionRelease\\00$x.txt")
        try source.mkString + " <SENTENCE_END>" finally source.close()
      }
    }

    val texts2 : Seq[String] = Seq.range(0, 9).map {
      x => {
        val source = scala.io.Source.fromFile(s"n-gram-language-model\\src\\main\\resources\\sentencePrediction\\textsForTraining\\shippingNotification\\00$x.txt")
        try source.mkString + " <SENTENCE_END>" finally source.close()
      }
    }


    while (true) {
      val input = readLine("Please type your text\n")

      val pipelineModel: PipelineModel = nlpPipeline.fit((texts ++ texts2).toDF("text"))

      val annotated: Map[String, Seq[Annotation]] = new LightPipeline(pipelineModel).fullAnnotate(input)

      println("\nSentence prediction: ")
      annotated("sentencePrediction").foreach(x => print(s"${x.result} "))
      
      print("\n\n")

    }
  }

}
