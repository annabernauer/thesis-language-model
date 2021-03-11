package com.mailytica.thesis.language.model.ngram.regexDemo

import com.johnsnowlabs.nlp.{Annotation, LightPipeline}
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.mailytica.thesis.language.model.ngram.regexDemo.WordLengthMatching.{getGeneralStages, getSpecificStages}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.scalatest.{Matchers, WordSpec}


class WordLengthMatchingSpec extends WordSpec with Matchers {

  "A text" when {
    ResourceHelper.spark

    import ResourceHelper.spark.implicits._

    val nlpPipeline = new Pipeline()

    nlpPipeline.setStages(getGeneralStages() ++ getSpecificStages())

    val pipelineModel: PipelineModel = nlpPipeline.fit(Seq.empty[String].toDF("text"))

    "has matches" should {

      val textWithMatches = """Quantum test million hundred test 123 12345 1234"""

      val annotated: Map[String, Seq[Annotation]] = new LightPipeline(pipelineModel).fullAnnotate(textWithMatches)
      annotated.foreach(println)

      "have the correct entries" in {
        annotated("matchedText").head.result should be("Quantum")
        annotated("matchedText").last.result should be("12345")
        annotated("matchedText").size should be(4)
      }
    }
  }
}
