package com.mailytica.thesis.language.model.ngram.matching

import com.johnsnowlabs.nlp.annotator.RegexMatcherModel
import org.apache.spark.ml.PipelineStage

object RegexMatching extends AbstractMatching {

  val regexRules: Array[String] = Array("""Quantum\s\w+""", """million\s\w+""", """John\s\w+, followed by leader""",
    """payment.*?\s""", """rall.*?\s""", """\d\d\d\d""", """\d+ Years""")

  override def getSpecificStages(): Array[_ <: PipelineStage] = {

    val regexMatcherModel = new RegexMatcherModel()
      .setInputCols("document")
      .setOutputCol("matchedText")
      .setRules(stringArrayToTupelArray(regexRules))
      .setStrategy("MATCH_ALL")

    Array(regexMatcherModel)

  }

  def stringArrayToTupelArray(stringArray: Array[String]): Array[(String, String)] =
    Array.range(0, stringArray.length).map(i => (stringArray(i), i.toString))

}
