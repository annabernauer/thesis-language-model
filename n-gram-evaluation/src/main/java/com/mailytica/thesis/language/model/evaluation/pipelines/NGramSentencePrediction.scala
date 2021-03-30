package com.mailytica.thesis.language.model.evaluation.pipelines

import com.johnsnowlabs.nlp.DocumentAssembler
import com.johnsnowlabs.nlp.annotator.Tokenizer
import com.mailytica.thesis.language.model.evaluation.annotators.ngram.NGramSentenceEvaluation
import com.mailytica.thesis.language.model.evaluation.annotators.{RedundantTextTrimmer, SentenceEndMarker, SentenceSplitter}
import org.apache.spark.ml.PipelineStage

object NGramSentencePrediction {

  def getStages(n: Int = 3): Array[_ <: PipelineStage] = {

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")
      .setCleanupMode("disabled")

    val redundantTextTrimmer = new RedundantTextTrimmer()
      .setInputCols("document")
      .setOutputCol("trimmedDocument")

    val sentenceSplitter = new SentenceSplitter()
      .setInputCols("trimmedDocument")
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

    Array(documentAssembler, redundantTextTrimmer, sentenceSplitter, markedSentenceEnds, tokenizer, nGramSentenceAnnotator)
  }

}
