package com.mailytica.thesis.language.model.ngram.pipelines.nGramSentences

import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.nlp.{Annotation, LightPipeline}
import com.mailytica.thesis.language.model.ngram.pipelines.nGramSentences.ExecutableSentencePrediction.getResourceText
import com.mailytica.thesis.language.model.ngram.pipelines.nGramSentences.NGramSentencePrediction.getStages
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.junit.runner.RunWith
import org.scalatest.{Matchers, WordSpec}
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class NGramSentencePredictionSpec extends WordSpec with Matchers {

  "A text" when {

    ResourceHelper.spark

    import ResourceHelper.spark.implicits._

    val nlpPipeline = new Pipeline()


    "is trained and has matches" when {
      nlpPipeline.setStages(getStages())

      val textWithMatches = """Quantum test million. million Quantum test million. test Quantum test"""

      val pipelineModel: PipelineModel = nlpPipeline.fit(Seq(textWithMatches).toDF("text"))

      "has a text with matches" should {

        val annotated: Map[String, Seq[Annotation]] = new LightPipeline(pipelineModel).fullAnnotate(textWithMatches)
        "have predicted the sentence" in {

          annotated("sentencePrediction")(12).result should be("million")
          annotated("sentencePrediction")(13).result should be(".")

        }

      }

      "have empty columns" in {

        val annotated: Map[String, Seq[Annotation]] = new LightPipeline(pipelineModel).fullAnnotate("")

        annotated("sentencePrediction").isEmpty should be (true)

      }
    }

    "is trained with an empty text" should {
      nlpPipeline.setStages(getStages())

      val textWithMatches = ""

      val pipelineModel: PipelineModel = nlpPipeline.fit(Seq(textWithMatches).toDF("text"))

      "have no predicted sentences" in {

        val annotated: Map[String, Seq[Annotation]] = new LightPipeline(pipelineModel).fullAnnotate(textWithMatches)

        annotated("sentencePrediction").lastOption should be(None)

        annotated("sentencesWithFlaggedEnds").headOption should be (None)
      }

      "have empty columns" in {

        val annotated: Map[String, Seq[Annotation]] = new LightPipeline(pipelineModel).fullAnnotate("")

        annotated("sentencePrediction").isEmpty should be (true)

      }
    }

    "is trained with big data" when {
      nlpPipeline.setStages(getStages(4))


      val texts : Seq[String] = getResourceText("/sentencePrediction/textsForTraining/productionRelease")

      val texts2 : Seq[String] = getResourceText("/sentencePrediction/textsForTraining/shippingNotification")

      val input : Seq[String] =
        Seq("Bitte klicken Sie", "Ihre Bestellung mit", "Der Versand erfolgt", "Bitte zögern Sie", "Wir bedanken uns", "Mit freundlichen Grüßen")

      val pipelineModel: PipelineModel = nlpPipeline.fit((texts ++ texts2).toDF("text"))

      "has a text with matches" should {

        val annotated: Seq[Map[String, Seq[Annotation]]] = input.map(inputString => new LightPipeline(pipelineModel).fullAnnotate(inputString))
        "have predicted the sentence" in {
          annotated(0)("sentencePrediction").length should be(16)
          annotated(2)("sentencePrediction").length should be(7)
          annotated(3)("sentencePrediction").length should be(9)
          annotated(4)("sentencePrediction").length should be(23)
          annotated(5)("sentencePrediction").length should be(3)
        }

      }
    }
  }
}
