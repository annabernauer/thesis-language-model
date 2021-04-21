package com.mailytica.thesis.language.model.ngram.pipelines

import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.nlp.{Annotation, LightPipeline}
import com.mailytica.thesis.language.model.ngram.pipelines.matching.RegexMatching.{getGeneralStages, getSpecificStages}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{Matchers, WordSpec}

@RunWith(classOf[JUnitRunner])
class RegexMatchingSpec extends WordSpec with Matchers {

  "A text" when {
    ResourceHelper.spark

    import ResourceHelper.spark.implicits._

    val nlpPipeline = new Pipeline()

    nlpPipeline.setStages(getGeneralStages() ++ getSpecificStages())

    val pipelineModel: PipelineModel = nlpPipeline.fit(Seq.empty[String].toDF("text"))

    "has matches" should {

      val textWithMatches = """Quantum test test million hundred test"""

      val annotated: Map[String, Seq[Annotation]] = new LightPipeline(pipelineModel).fullAnnotate(textWithMatches)
      annotated.foreach(println)

      "have the correct entries" in {
        annotated("regex").head.result should be("Quantum test")
        annotated("regex").last.result should be("million hundred")
        annotated("regex").size should be(2)
      }
    }
    "has no matches" should {

      val textWithoutMatch = """1 2 3"""

      val annotated: Map[String, Seq[Annotation]] = new LightPipeline(pipelineModel).fullAnnotate(textWithoutMatch)
      println("")
      annotated.foreach(println)

      "have size 0" in {
        annotated.get("regex").head.size should be(0)
      }
    }

    "is empty" should {

      val emptyText = ""

      val annotated: Map[String, Seq[Annotation]] = new LightPipeline(pipelineModel).fullAnnotate(emptyText)
      println("")
      annotated.foreach(println)

      "have size 0" in {
        annotated.get("regex").head.size should be(0)
      }
    }
  }
}
