package com.mailytica.thesis.language.model.ngram.matching

import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.nlp.{Annotation, LightPipeline}
import com.mailytica.thesis.language.model.ngram.matching.NGramProbability.{getGeneralStages, getSpecificStages}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{Matchers, WordSpec}

@RunWith(classOf[JUnitRunner])
class NGramProbabilitySpec extends WordSpec with Matchers {

  "A text" when {
    ResourceHelper.spark

    import ResourceHelper.spark.implicits._

    val nlpPipeline = new Pipeline()

    val textWithMatches = """Quantum test million test million Quantum test million test Quantum test million"""

    nlpPipeline.setStages(getGeneralStages() ++ getSpecificStages())

    val pipelineModel: PipelineModel = nlpPipeline.fit(Seq(textWithMatches).toDF("text"))

    "has entries" should {

      val annotated: Map[String, Seq[Annotation]] = new LightPipeline(pipelineModel).fullAnnotate(textWithMatches)

      print("\nin Testclass\n")
      annotated.foreach(println)

      "have a prediction" in {

        val annotation = annotated("ngramsProbability").head
        annotation.result should be("test")
        annotation.end - annotation.begin should be(4)

        annotation.metadata.getOrElse("probability", "no probability") should be("0.5")

      }
    }

    "has no entries" should {

      val annotated: Map[String, Seq[Annotation]] = new LightPipeline(pipelineModel).fullAnnotate("")

      "have empty columns" in {

        annotated("ngramsProbability").isEmpty should be (true)

      }
    }

  }
}