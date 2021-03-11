package com.mailytica.thesis.language.model.ngram.matching

import java.io.{BufferedWriter, File, FileWriter}

import com.johnsnowlabs.nlp.annotators.TextMatcher
import org.apache.spark.ml.PipelineStage

object TextMatching extends AbstractMatching {

  val exactMatches: Array[String] = Array("Quantum", "million", "payments", "index", "market share", "gap", "market",
    "measure", "aspects", "accounts", "king")

  override def getSpecificStages(): Array[_ <: PipelineStage] = {

    writeToFile("n-gram-language-model\\target\\textToMatch.txt", exactMatches)

    val textMatcher = new TextMatcher()
      .setInputCols(Array("document", "token"))
      .setOutputCol("matchedText")
      .setCaseSensitive(false)
      .setEntities("n-gram-language-model\\target\\textToMatch.txt", "TEXT")

    Array(textMatcher)

  }

  def writeToFile(pathname: String, arrayToWrite: Array[String]): File = {

    val file: File = new File(pathname)
    val bw = new BufferedWriter(new FileWriter(file))

    for (regexRule <- arrayToWrite) {
      bw.write(regexRule)
      bw.write("\n")
    }

    bw.close()
    file

  }

}
