package com.mailytica.thesis.language.model.evaluation.annotators.ngram

import breeze.numerics.{log, sqrt}
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.nlp.{Annotation, LightPipeline}
import com.mailytica.thesis.language.model.evaluation.pipelines.NGramSentencePrediction.getStages
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.junit.runner.RunWith
import org.scalatest.{Matchers, WordSpec}
import org.scalatest.junit.JUnitRunner

import scala.io.{Codec, Source}

@RunWith(classOf[JUnitRunner])
class NGramSentenceEvaluationSpec extends WordSpec with Matchers {

  "A text" when {

    ResourceHelper.spark

    import ResourceHelper.spark.implicits._

    val nlpPipeline = new Pipeline()


    "is trained with big data" when {
      nlpPipeline.setStages(getStages(4))


      val texts: Seq[String] = getResourceText("/sentencePrediction/textsForTraining/productionRelease")

      val texts2: Seq[String] = getResourceText("/sentencePrediction/textsForTraining/shippingNotification")

      val pipelineModel: PipelineModel = nlpPipeline.fit((texts ++ texts2).toDF("text"))

      "has a text with matches" should {

        val annotated: Seq[Map[String, Seq[Annotation]]] = texts.map(inputString => new LightPipeline(pipelineModel).fullAnnotate(inputString))
//        val annotated: Map[String, Seq[Annotation]] = new LightPipeline(pipelineModel).fullAnnotate(texts.head)

        val flatMap : Seq[Annotation] = annotated.flatMap(map => map("sentencePrediction"))

        val likelihoods: Seq[Double] =
          flatMap
          .map(annotation => annotation.metadata.getOrElse("probability", "0.0").toDouble)

        val invertedLikelihoods: Seq[Double] = likelihoods.map(likelihood => 1/likelihood)
        val perplexity : Double = sqrt(invertedLikelihoods.foldLeft(1.0)(_ * _))

        val avgLogLikelihood: Double =
          likelihoods
            .map(likelihood => log(likelihood))
            .foldLeft(0.0)(_ + _) / likelihoods.size

        flatMap.foreach(annotation => println(s"${annotation.result} ${annotation.metadata}"))
        println(perplexity)
        println(avgLogLikelihood)

        "have predicted the sentence" in {

        }

      }
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
