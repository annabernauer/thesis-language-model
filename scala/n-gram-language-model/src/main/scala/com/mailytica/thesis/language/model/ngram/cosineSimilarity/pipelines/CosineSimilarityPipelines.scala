package com.mailytica.thesis.language.model.ngram.cosineSimilarity.pipelines

import com.johnsnowlabs.nlp.annotator.Tokenizer
import com.johnsnowlabs.nlp.{DocumentAssembler, Finisher}
import com.mailytica.thesis.language.model.ngram.annotators.ngram.{NGramCustomGenerator, NGramSentenceAnnotator}
import com.mailytica.thesis.language.model.ngram.annotators.{RedundantTextTrimmer, SentenceEndMarker, SentenceSplitter}
import com.mailytica.thesis.language.model.ngram.cosineSimilarity.annotators.{ExplodedTransformer, SentenceNewLineRemover, SentenceSeedExtractor, TokensMerger}
import org.apache.spark.ml.PipelineStage
import org.apache.spark.ml.feature.HashingTF

object CosineSimilarityPipelines {


  def getPreprocessStages(n: Int = 3): Array[_ <: PipelineStage] = {

    val documentAssembler = new DocumentAssembler()
      .setInputCol("data")
      .setOutputCol("document")
      .setCleanupMode("disabled")
    //timer(dupl)
    val redundantTextTrimmer = new RedundantTextTrimmer()
      .setInputCols(documentAssembler.getOutputCol)
      .setOutputCol("trimmedDocument")
    //timer(dupl)
    val sentenceSplitter = new SentenceSplitter()
      .setInputCols(redundantTextTrimmer.getOutputCol)
      .setOutputCol("sentences")

    val markedSentenceEnds = new SentenceEndMarker()
      .setInputCols("sentences")
      .setOutputCol("sentencesWithFlaggedEnds")

    val finisher = new Finisher()
      .setInputCols(markedSentenceEnds.getOutputCol)
      .setOutputCols("finishedSentences")
      .setIncludeMetadata(false)

    val explodedTransformer = new ExplodedTransformer() //one column per sentence, references
      .setInputCol(finisher.getOutputCols.head)
      .setOutputCol("referenceSentences")

    val documentAssemblerDue = new DocumentAssembler()
      .setInputCol(explodedTransformer.getOutputCol)
      .setOutputCol("explodedDocument")
      .setCleanupMode("disabled")

    val tokenizer = new Tokenizer()
      .setInputCols(documentAssemblerDue.getOutputCol)
      .setOutputCol("token")
    //timer
    val sentenceSeed = new SentenceSeedExtractor()
      .setInputCols(tokenizer.getOutputCol)
      .setOutputCol("seeds")
      .setN(n)

    val finisherDue = new Finisher()
      .setInputCols(sentenceSeed.getOutputCol)
      .setOutputCols("seeds")
      .setOutputAsArray(false)

    Array(documentAssembler, redundantTextTrimmer, sentenceSplitter, markedSentenceEnds, finisher, explodedTransformer, documentAssemblerDue, tokenizer, sentenceSeed, finisherDue)
  }

  def getVectorizerStages(inputCol: String, identifier: String, needsDocAssembl: Boolean): Array[_ <: PipelineStage] = {

    val (documentAssemblerOpt: Option[DocumentAssembler], tokenizer: Tokenizer) = needsDocAssembl match {
      case true =>
        val documentAssembler: DocumentAssembler = new DocumentAssembler()
          .setInputCol(inputCol)
          .setOutputCol("document_" + identifier)
          .setCleanupMode("disabled")

        val tokenizer = new Tokenizer()
          .setInputCols(documentAssembler.getOutputCol)
          .setOutputCol("tokens_" + identifier)
        (Option(documentAssembler), tokenizer)
      case false =>
        val tokenizer = new Tokenizer()
          .setInputCols(inputCol)
          .setOutputCol("tokens_" + identifier)
        (None, tokenizer)
    }

    val nGramCustomGenerator = new NGramCustomGenerator()
      .setInputCols(tokenizer.getOutputCol)
      .setOutputCol("ngrams_" + identifier)
      .setN(3)
      .setNGramMinimum(1) //TODO from 1 or 3? ngrams in evaluation

    val finisher = new Finisher()
      .setInputCols(nGramCustomGenerator.getOutputCol)
      .setOutputCols("finishedNgrams_" + identifier)
      .setCleanAnnotations(false)

    val countVector = new HashingTF()
      .setInputCol(finisher.getOutputCols.head)
      .setOutputCol("vectorizedCount_" + identifier)

    documentAssemblerOpt match {
      case Some(documentAssembler) => Array(documentAssembler, tokenizer, nGramCustomGenerator, finisher, countVector)
      case None => Array(tokenizer, nGramCustomGenerator, finisher, countVector)
    }
  }

  def getPredictionStages(n: Int = 3): Array[_ <: PipelineStage] = {

    //context processing
    val documentAssembler = new DocumentAssembler()
      .setInputCol("seeds")
      .setOutputCol("document")
      .setCleanupMode("disabled")
    //timer
    val redundantTextTrimmer = new RedundantTextTrimmer()
      .setInputCols("document")
      .setOutputCol("trimmedDocument")

    //timer, sentencesplitter and endmarker mergen?
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

    //timer
    val sentenceNewLineRemover = new SentenceNewLineRemover()
      .setInputCols("referenceDocument")
      .setOutputCol("referenceWithoutNewLines")

    Array(documentAssemblerReference, sentenceNewLineRemover)
  }

}
