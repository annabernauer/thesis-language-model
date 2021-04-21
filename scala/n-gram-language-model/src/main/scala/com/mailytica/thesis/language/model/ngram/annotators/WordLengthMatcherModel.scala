package com.mailytica.thesis.language.model.ngram.annotators

import com.johnsnowlabs.nlp.AnnotatorType.TOKEN
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, AnnotatorType}
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.Identifiable

class WordLengthMatcherModel(override val uid: String) extends AnnotatorModel[WordLengthMatcherModel]{

  override val inputAnnotatorTypes: Array[String] = Array(TOKEN)

  override val outputAnnotatorType: AnnotatorType = TOKEN

  def this() = this(Identifiable.randomUID("WORD_LENGTH_MATCHER"))

  val wordLength: Param[Int] = new Param(this, "word length", "")

  def setWordLength(wordLength: Int): WordLengthMatcherModel = set(this.wordLength, wordLength)

  setDefault(this.wordLength, 4)

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {

    val tokens = annotations
      .filter(token => token.annotatorType == AnnotatorType.TOKEN)
      .filter(token => token.result.length > $(wordLength))

    tokens
  }
}
