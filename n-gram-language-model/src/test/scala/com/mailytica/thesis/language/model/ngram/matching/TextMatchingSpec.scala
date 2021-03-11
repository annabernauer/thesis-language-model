package com.mailytica.thesis.language.model.ngram.matching

import com.johnsnowlabs.nlp.{Annotation, LightPipeline}
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.mailytica.thesis.language.model.ngram.matching.TextMatching.{getGeneralStages, getSpecificStages}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.scalatest.{Matchers, WordSpec}

class TextMatchingSpec extends WordSpec with Matchers{

  "A text" when {
    ResourceHelper.spark

    import ResourceHelper.spark.implicits._

    val nlpPipeline = new Pipeline()

    nlpPipeline.setStages(getGeneralStages() ++ getSpecificStages())

    val pipelineModel: PipelineModel = nlpPipeline.fit(Seq.empty[String].toDF("text"))

    "has matches" should {

      val textWithMatches = """Quantum test million hundred test"""

      val annotated: Map[String, Seq[Annotation]] = new LightPipeline(pipelineModel).fullAnnotate(textWithMatches)
      annotated.foreach(println)

      "have the correct entries" in {
        annotated("matchedText").head.result should be("Quantum")
        annotated("matchedText").last.result should be("million")
        annotated("matchedText").size should be(2)
      }
    }
    "has no matches" should {

      val textWithoutMatch = """1 2 3"""

      val annotated: Map[String, Seq[Annotation]] = new LightPipeline(pipelineModel).fullAnnotate(textWithoutMatch)
      println("")
      annotated.foreach(println)

      "have size 0" in {
        annotated.get("matchedText").head.size should be(0)
      }
    }

    "is empty" should {

      val emptyText = ""

      val annotated: Map[String, Seq[Annotation]] = new LightPipeline(pipelineModel).fullAnnotate(emptyText)
      println("")
      annotated.foreach(println)

      "have size 0" in {
        annotated.get("matchedText").head.size should be(0)
      }
    }
  }
}
