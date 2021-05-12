package com.mailytica.thesis.language.model.ngram.cosineSimilarity

import com.johnsnowlabs.nlp.{DocumentAssembler, Finisher}
import com.johnsnowlabs.nlp.annotator.Tokenizer
import com.mailytica.thesis.language.model.ngram.annotators.{RedundantTextTrimmer, SentenceEndMarker, SentenceSplitter}
import com.mailytica.thesis.language.model.ngram.annotators.ngram.NGramSentenceAnnotator
import com.mailytica.thesis.language.model.ngram.cosineSimilarity.annotators.{ExplodedTransformer, SentenceSeedExtractor}
import org.apache.spark.ml.PipelineStage
import org.apache.spark.sql.catalyst.expressions.Explode

object PreprocessTestDataPipeline {

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

    val markedSentenceEnds = new SentenceEndMarker()
      .setInputCols("sentences")
      .setOutputCol("sentencesWithFlaggedEnds")

    val finisher = new Finisher()
      .setInputCols("sentencesWithFlaggedEnds")
      .setOutputCols("finishedSentences")
      .setIncludeMetadata(true)

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

    Array(documentAssembler, redundantTextTrimmer, sentenceSplitter, markedSentenceEnds, finisher, explodedTransformer,
      documentAssemblerDue, tokenizer, sentenceSeed)
  }

}
