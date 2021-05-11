package com.mailytica.thesis.language.model.evaluationLM.pipelines

import com.johnsnowlabs.nlp.DocumentAssembler
import com.johnsnowlabs.nlp.annotator.Tokenizer
import com.mailytica.thesis.language.model.evaluationLM.annotators.ngram.NGramSentenceEvaluation
import com.mailytica.thesis.language.model.evaluationLM.annotators.{RedundantTextTrimmer, SentenceEndMarker, SentenceSplitter, ShortDocumentsFilter}
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

//    val markedSentenceEnds = new SentenceEndMarker()
//      .setInputCols("sentences")
//      .setOutputCol("sentencesWithFlaggedEnds")

    val tokenizer = new Tokenizer()
      .setInputCols("sentences")
      .setOutputCol("token")

    val shortDocumentsFilter = new ShortDocumentsFilter()
      .setInputCols("token")
      .setOutputCol("tokenFiltered")

    val nGramSentenceAnnotator = new NGramSentenceEvaluation()
      .setInputCols("tokenFiltered")
      .setOutputCol("sentencePrediction")
      .setN(n)

    Array(documentAssembler, redundantTextTrimmer, sentenceSplitter, tokenizer, shortDocumentsFilter, nGramSentenceAnnotator)
//    Array(documentAssembler, redundantTextTrimmer, sentenceSplitter, tokenizer)
  }

}
