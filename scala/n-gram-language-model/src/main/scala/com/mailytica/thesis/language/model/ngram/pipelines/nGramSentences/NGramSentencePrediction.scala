package com.mailytica.thesis.language.model.ngram.pipelines.nGramSentences

import com.johnsnowlabs.nlp.DocumentAssembler
import com.johnsnowlabs.nlp.annotator.{SentenceDetector, Tokenizer}
import com.mailytica.thesis.language.model.ngram.annotators.ngram.NGramSentenceAnnotator
import com.mailytica.thesis.language.model.ngram.annotators.{RedundantTextTrimmer, SentenceEndMarker, SentenceSplitter}
import com.mailytica.thesis.language.model.ngram.cosineSimilarity.annotators.TokensMerger
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

    val nGramSentenceAnnotator = new NGramSentenceAnnotator()
      .setInputCols("token")
      .setOutputCol("sentencePrediction")
      .setN(n)

    val tokensMerger = new TokensMerger()
      .setInputCols("sentencePrediction")
      .setOutputCol("mergedPrediction")

    Array(documentAssembler, redundantTextTrimmer, sentenceSplitter, markedSentenceEnds, tokenizer, nGramSentenceAnnotator, tokensMerger)
  }

}
