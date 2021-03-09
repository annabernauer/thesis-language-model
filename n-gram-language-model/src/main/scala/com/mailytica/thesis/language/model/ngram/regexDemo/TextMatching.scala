package com.mailytica.thesis.language.model.ngram.regexDemo

import com.johnsnowlabs.nlp.annotators.TextMatcher
import org.apache.spark.ml.PipelineStage

object TextMatching extends AbstractMatching {

  def main(args: Array[String]): Unit = {

    val exactMatches = Array("Quantum", "million", "payments", "index", "market share", "gap", "market",
      "measure", "aspects", "accounts", "king")

    val file = writeToFile("textToMatch.txt", exactMatches)

    val nlpData = createDataframe()

    nlpData.select("matchedText").show(false)

    file.delete()

  }

  override def getSpecificStages(): Array[_ <: PipelineStage] = {

    val textMatcher = new TextMatcher()
      .setInputCols(Array("document", "token"))
      .setOutputCol("matchedText")
      .setCaseSensitive(false)
      .setEntities("textToMatch.txt", "TEXT")

    Array(textMatcher)

  }

}
