package com.mailytica.thesis.language.model.ngram.matching

import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.nlp.{Annotation, LightPipeline}
import com.mailytica.thesis.language.model.ngram.matching.NGramProbability.{getGeneralStages, getSpecificStages}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.scalatest.{Matchers, WordSpec}

class NGramProbabilitySpec extends WordSpec with Matchers {
  ResourceHelper.spark

  import ResourceHelper.spark.implicits._

  val nlpPipeline = new Pipeline()

  val textWithMatches = """Quantum test million test million Quantum test million test Quantum test million"""

  nlpPipeline.setStages(getGeneralStages() ++ getSpecificStages())

  val pipelineModel: PipelineModel = nlpPipeline.fit(Seq(textWithMatches).toDF("text"))

  "has matches" should {

    val annotated: Map[String, Seq[Annotation]] = new LightPipeline(pipelineModel).fullAnnotate(textWithMatches)

    print("\nin Testclass\n")
    annotated.foreach(println)

    "have the correct entries" in {
      annotated("ngramsProbability").head.result should be("test")
//      annotated("ngrams").last.result should be("test_million")
//      annotated("ngrams").size should be(5)
    }
  }
}