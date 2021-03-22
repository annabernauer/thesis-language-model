package com.mailytica.thesis.language.model.ngram.nGramSentences

import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.nlp.{Annotation, LightPipeline}
import com.mailytica.thesis.language.model.ngram.nGramSentences.NGramSentencePrediction.getStages
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

    nlpPipeline.setStages(getStages())

    "is trained and has matches" when {

      val textWithMatches = """Quantum test million. million Quantum test million. test Quantum test"""

      val pipelineModel: PipelineModel = nlpPipeline.fit(Seq(textWithMatches).toDF("text"))

      "has a text with matches" should {

        val annotated: Map[String, Seq[Annotation]] = new LightPipeline(pipelineModel).fullAnnotate(textWithMatches)
        "have predicted the sentence" in {

          print("\nin Testclass\n")
          annotated("sentencePrediction").foreach(println)

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
  }
}
