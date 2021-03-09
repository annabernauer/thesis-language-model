package com.mailytica.thesis.language.model.ngram.regexDemo

import com.johnsnowlabs.nlp.annotator.{RegexMatcher, RegexMatcherModel}
import org.apache.spark.ml.PipelineStage
import org.apache.spark.sql.DataFrame

object RegexMatching extends AbstractMatching {

  val regexRules: Array[String] = Array("""Quantum\s\w+""", """million\s\w+""", """John\s\w+, followed by leader""",
    """payment.*?\s""", """rall.*?\s""", """\d\d\d\d""", """\d+ Years""")

  def main(args: Array[String]): Unit = {

    val nlpData: DataFrame = createDataframe()

    nlpData.select("matchedText").show(false)

    //  val annotated: Map[String, Seq[Annotation]] = new LightPipeline(pipelineModel).fullAnnotate(textList.head)
    //  annotated.foreach(println)

    //  annotated.get("document").head.equals(Annotation("document", 0, 9, "result..."))

  }

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
