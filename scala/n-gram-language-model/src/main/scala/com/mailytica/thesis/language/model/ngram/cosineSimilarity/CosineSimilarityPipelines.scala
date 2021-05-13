package com.mailytica.thesis.language.model.ngram.cosineSimilarity

import com.johnsnowlabs.nlp.{DocumentAssembler, Finisher}
import com.johnsnowlabs.nlp.annotator.Tokenizer
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.mailytica.thesis.language.model.ngram.annotators.{RedundantTextTrimmer, SentenceEndMarker, SentenceSplitter}
import com.mailytica.thesis.language.model.ngram.annotators.ngram.NGramSentenceAnnotator
import com.mailytica.thesis.language.model.ngram.cosineSimilarity.annotators.{ExplodedTransformer, SentenceNewLineRemover, SentenceSeedExtractor, TokensMerger}
import org.apache.spark.ml.PipelineStage
import org.apache.spark.ml.feature.CountVectorizer
import org.apache.spark.sql.catalyst.expressions.Explode

object CosineSimilarityPipelines {

  def getPreprocessStages(n: Int = 3): Array[_ <: PipelineStage] = {

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



    val finisher = new Finisher()
      .setInputCols("sentences")
      .setOutputCols("finishedSentences")
      .setIncludeMetadata(false)

    val explodedTransformer = new ExplodedTransformer()
      .setInputCol("finishedSentences")
      .setOutputCol("explodedSentences")

    val documentAssemblerDue = new DocumentAssembler()
      .setInputCol("explodedSentences")
      .setOutputCol("explodedDocument")
      .setCleanupMode("disabled")

    val tokenizer = new Tokenizer()
      .setInputCols("explodedDocument")
      .setOutputCol("token")

    val sentenceSeed = new SentenceSeedExtractor()
      .setInputCols("token")
      .setOutputCol("seeds")
      .setN(n)

    Array(documentAssembler, redundantTextTrimmer, sentenceSplitter, finisher, explodedTransformer, documentAssemblerDue, tokenizer, sentenceSeed)
  }

  def getVectorizerStages: Array[_ <: PipelineStage] = {

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")
      .setCleanupMode("disabled")

    val sentenceDetector = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentences")

    val tokenizer = new Tokenizer()
      .setInputCols("sentences")
      .setOutputCol("tokens")

    val finisher = new Finisher()
      .setInputCols("tokens")
      .setOutputCols("finishedTokens")

    val countVector = new CountVectorizer()
      .setInputCol("finishedTokens")
      .setOutputCol("vectorizedCount")

    Array(documentAssembler, sentenceDetector, tokenizer, finisher, countVector)
  }

  def getPredictionStages(n: Int = 3): Array[_ <: PipelineStage] = {

    //context processing
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

  def getReferenceStages(): Array[_ <: PipelineStage] = {
        //reference processing -> removing of new lines
        val documentAssemblerReference = new DocumentAssembler()
          .setInputCol("reference")
          .setOutputCol("referenceDocument")
          .setCleanupMode("disabled")

        val sentenceNewLineRemover = new SentenceNewLineRemover()
          .setInputCols("referenceDocument")
          .setOutputCol("referenceWithoutNewLines")

    Array(documentAssemblerReference, sentenceNewLineRemover)
  }

}
