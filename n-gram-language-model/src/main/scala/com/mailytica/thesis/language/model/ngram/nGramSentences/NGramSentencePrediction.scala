package com.mailytica.thesis.language.model.ngram.nGramSentences

import com.johnsnowlabs.nlp.DocumentAssembler
import com.johnsnowlabs.nlp.annotator.{SentenceDetector, Tokenizer}
import com.mailytica.thesis.language.model.ngram.annotator.{NGramSentenceAnnotator, SentenceEndMarker, SentenceSplitter}
import org.apache.spark.ml.PipelineStage


object NGramSentencePrediction {

  def getStages(n: Int = 3): Array[_ <: PipelineStage] = {

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")
      .setCleanupMode("disabled")

//    val sentenceSplitter = new SentenceDetector()
//      .setInputCols("document")
//      .setOutputCol("sentences")
////      .setCustomBounds(Array("\\:", "\\R"))
//      .setCustomBounds(Array("\\R"))

        val sentenceSplitter = new SentenceSplitter()
          .setInputCols("document")
          .setOutputCol("sentences")
    ////      .setCustomBounds(Array("\\:", "\\R"))
    //      .setCustomBounds(Array("\\R"))

    val markedSentenceEnds = new SentenceEndMarker()
      .setInputCols("sentences")
      .setOutputCol("sentencesWithFlaggedEnds")

    val tokenizer = new Tokenizer()
      .setInputCols("sentencesWithFlaggedEnds")
      .setOutputCol("token")

    val nGramSentenceAnnotator = new NGramSentenceAnnotator()
      .setInputCols("token")
      .setOutputCol("sentencePrediction")
      .setN(n)

    Array(documentAssembler, sentenceSplitter, markedSentenceEnds, tokenizer, nGramSentenceAnnotator)
  }

}
