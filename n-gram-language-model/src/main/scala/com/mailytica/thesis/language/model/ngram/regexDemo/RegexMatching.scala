package com.mailytica.thesis.language.model.ngram.regexDemo

import com.johnsnowlabs.nlp.annotator.RegexMatcher
import org.apache.spark.ml.PipelineStage
import org.apache.spark.sql.DataFrame

object RegexMatching extends AbstractMatching {

  def main(args: Array[String]): Unit = {

    val regexRules: Array[String] = Array("""Quantum\s\w+""", """million\s\w+""", """John\s\w+, followed by leader""",
      """payment.*?\s""", """rall.*?\s""", """\d\d\d\d""", """\d+ Years""")

    val file = writeToFile("regexToMatch.txt", regexRules)

    val nlpData: DataFrame = createDataframe()

    nlpData.select("matchedText").show(false)

    file.delete()

    //  val annotated: Map[String, Seq[Annotation]] = new LightPipeline(pipelineModel).fullAnnotate(textList.head)
    //  annotated.foreach(println)

    //  annotated.get("document").head.equals(Annotation("document", 0, 9, "result..."))

  }

  override def getSpecificStages(): Array[_ <: PipelineStage] = {

    val regexMatcher: RegexMatcher = new RegexMatcher()
      .setInputCols("document")
      .setOutputCol("matchedText")
      .setStrategy("MATCH_ALL")
      .setRules("regexToMatch.txt", ",")

    //    val regexMatcherModel = new RegexMatcherModel()
    //      .setInputCols("document")
    //      .setOutputCol("matchedTextWithModel")
    //      .setRules("")

    Array(regexMatcher)

  }
}
