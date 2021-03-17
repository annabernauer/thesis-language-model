package com.mailytica.thesis.language.model.ngram.matching

import com.johnsnowlabs.nlp.{Annotation, LightPipeline}
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.mailytica.thesis.language.model.ngram.matching.NGramPropability.{getGeneralStages, getSpecificStages}
import com.mailytica.thesis.language.model.ngram.textSplittingDemo.TextSplitting.{data, pipeline}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.scalatest.{Matchers, WordSpec}

class NGramPropabilitySpec extends WordSpec with Matchers {
  ResourceHelper.spark

  import ResourceHelper.spark.implicits._

  val nlpPipeline = new Pipeline()

  val textWithMatches = """Quantum test million test million Quantum test million test Quantum test million"""
  val textWithMatches2 = """Quantum test"""

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