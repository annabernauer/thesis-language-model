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
      .setInputCol("data")
      .setOutputCol("document")
      .setCleanupMode("disabled")

    val redundantTextTrimmer = new RedundantTextTrimmer()
      .setInputCols(documentAssembler.getOutputCol)
      .setOutputCol("trimmedDocument")

    val sentenceSplitter = new SentenceSplitter()
      .setInputCols(redundantTextTrimmer.getOutputCol)
      .setOutputCol("sentences")

    val finisher = new Finisher()
      .setInputCols(sentenceSplitter.getOutputCol)
      .setOutputCols("finishedSentences")
      .setIncludeMetadata(false)

    val explodedTransformer = new ExplodedTransformer()         //one column per sentence, references
      .setInputCol(finisher.getOutputCols.head)
      .setOutputCol("referenceSentences")

    val documentAssemblerDue = new DocumentAssembler()
      .setInputCol(explodedTransformer.getOutputCol)
      .setOutputCol("explodedDocument")
      .setCleanupMode("disabled")

    val tokenizer = new Tokenizer()
      .setInputCols(documentAssemblerDue.getOutputCol)
      .setOutputCol("token")

    val sentenceSeed = new SentenceSeedExtractor()
      .setInputCols(tokenizer.getOutputCol)
      .setOutputCol("seeds")
      .setN(n)

    val finisherDue = new Finisher()
      .setInputCols(sentenceSeed.getOutputCol)
      .setOutputCols("seeds")
      .setOutputAsArray(false)

    Array(documentAssembler, redundantTextTrimmer, sentenceSplitter, finisher, explodedTransformer, documentAssemblerDue, tokenizer, sentenceSeed, finisherDue)
  }

//  def getVectorizerStages(inputCol: String, outputCol: String): Array[_ <: PipelineStage] = {
  def getVectorizerStages: Array[_ <: PipelineStage] = {

    val documentAssembler = new DocumentAssembler()
//      .setInputCol(inputCol)
//      .setOutputCol("document_" + inputCol)
      .setInputCol("mergedPrediction")
      .setOutputCol("document")
      .setCleanupMode("disabled")

    val sentenceDetector = new SentenceDetector()
      .setInputCols(documentAssembler.getOutputCol)
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
//      .setOutputCol(outputCol)

    Array(documentAssembler, sentenceDetector, tokenizer, finisher, countVector)
  }

  def getPredictionStages(n: Int = 3): Array[_ <: PipelineStage] = {

    //context processing
    val documentAssembler = new DocumentAssembler()
      .setInputCol("seeds")
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
          .setInputCol("referenceSentences")
          .setOutputCol("referenceDocument")
          .setCleanupMode("disabled")

        val sentenceNewLineRemover = new SentenceNewLineRemover()
          .setInputCols("referenceDocument")
          .setOutputCol("referenceWithoutNewLines")

    Array(documentAssemblerReference, sentenceNewLineRemover)
  }

}
