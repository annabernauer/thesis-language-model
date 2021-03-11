package com.mailytica.thesis.language.model.ngram.Annotator

import com.johnsnowlabs.nlp.AnnotatorType.{CHUNK, DOCUMENT}
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, AnnotatorType}
import org.apache.spark.ml.util.Identifiable

class WordLengthMatcherModel(override val uid: String) extends AnnotatorModel[WordLengthMatcherModel]{

  override val outputAnnotatorType: AnnotatorType = CHUNK

  override val inputAnnotatorTypes: Array[String] = Array(DOCUMENT)

  def this() = this(Identifiable.randomUID("WORD_LENGTH_MATCHER"))

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {

    val tokens = annotations.filter(token =>
      token.annotatorType == AnnotatorType.TOKEN &&
        (token.end - token.begin) >= 4)

    tokens
  }
}
