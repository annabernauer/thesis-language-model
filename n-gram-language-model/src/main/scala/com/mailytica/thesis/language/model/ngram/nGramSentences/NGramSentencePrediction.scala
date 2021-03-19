package com.mailytica.thesis.language.model.ngram.nGramSentences

import com.johnsnowlabs.nlp.DocumentAssembler
import com.johnsnowlabs.nlp.annotator.{SentenceDetector, Tokenizer}
import com.mailytica.thesis.language.model.ngram.annotator.{NGramSentenceAnnotator, SentenceEndMarker}
import org.apache.spark.ml.PipelineStage


object NGramSentencePrediction {

  def getStages(): Array[_ <: PipelineStage] = {

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentenceSplitter = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentences")

    val markedSentenceEnds = new SentenceEndMarker()
      .setInputCols("sentences")
      .setOutputCol("sentencesWithFlaggedEnds")

    val tokenizer = new Tokenizer()
      .setInputCols("sentencesWithFlaggedEnds")
      .setOutputCol("token")

    val nGramSentenceAnnotator = new NGramSentenceAnnotator()
      .setInputCols("token")
      .setOutputCol("sentencePrediction")

    Array(documentAssembler, sentenceSplitter, markedSentenceEnds, tokenizer, nGramSentenceAnnotator)
  }

}
