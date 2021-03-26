package com.mailytica.thesis.language.model.evaluation.pipelines

import com.johnsnowlabs.nlp.DocumentAssembler
import com.johnsnowlabs.nlp.annotator.Tokenizer
import com.mailytica.thesis.language.model.evaluation.annotators.ngram.NGramSentenceEvaluation
import com.mailytica.thesis.language.model.evaluation.annotators.{SentenceEndMarker, SentenceSplitter}
import org.apache.spark.ml.PipelineStage

object NGramSentencePrediction {

  def getStages(n: Int = 3): Array[_ <: PipelineStage] = {

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")
      .setCleanupMode("disabled")

    val sentenceSplitter = new SentenceSplitter()
      .setInputCols("document")
      .setOutputCol("sentences")

    val markedSentenceEnds = new SentenceEndMarker()
      .setInputCols("sentences")
      .setOutputCol("sentencesWithFlaggedEnds")

    val tokenizer = new Tokenizer()
      .setInputCols("sentencesWithFlaggedEnds")
      .setOutputCol("token")

    val nGramSentenceAnnotator = new NGramSentenceEvaluation()
      .setInputCols("token")
      .setOutputCol("sentencePrediction")
      .setN(n)

    Array(documentAssembler, sentenceSplitter, markedSentenceEnds, tokenizer, nGramSentenceAnnotator)
  }

}
