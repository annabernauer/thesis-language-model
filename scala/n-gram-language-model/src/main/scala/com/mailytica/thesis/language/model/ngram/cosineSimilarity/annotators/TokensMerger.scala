package com.mailytica.thesis.language.model.ngram.cosineSimilarity.annotators

import com.johnsnowlabs.nlp.AnnotatorType.{CHUNK, DOCUMENT, TOKEN}
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel}
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}

class TokensMerger(override val uid: String) extends AnnotatorModel[TokensMerger] with DefaultParamsWritable {
  override val inputAnnotatorTypes: Array[String] = Array(TOKEN)

  override val outputAnnotatorType: AnnotatorType = DOCUMENT

  def this() = this(Identifiable.randomUID("TOKEN_MERGER"))

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {

    val mergedTokens = annotations.map(annotation => annotation.result).mkString(" ")

    Seq(Annotation(
      annotatorType = DOCUMENT,
      begin = 0,
      end = mergedTokens.length,
      result = mergedTokens,
      metadata =  Map.empty
    ))
  }
}