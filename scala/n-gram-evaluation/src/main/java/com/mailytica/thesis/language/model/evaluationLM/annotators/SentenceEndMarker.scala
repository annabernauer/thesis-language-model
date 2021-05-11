package com.mailytica.thesis.language.model.evaluationLM.annotators

import com.johnsnowlabs.nlp.AnnotatorType.DOCUMENT
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel}
import org.apache.spark.ml.util.Identifiable

import scala.util.matching.Regex
//is replaced by SentenceSplitter
class SentenceEndMarker(override val uid: String) extends AnnotatorModel[SentenceEndMarker] {

  override val outputAnnotatorType: AnnotatorType = DOCUMENT

  override val inputAnnotatorTypes: Array[String] = Array(DOCUMENT)

  def this() = this(Identifiable.randomUID("SENTENCE_END_MARKER"))

  val SENTENCE_END: String = " <SENTENCE_END>"

  val SENTENCE_START: String = "SENTENCE_START> "

  val REGEX_SENTENCE_END: Regex = "(\\.|\\!|\\?|\\:|\\R)$".r

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {

    annotations
      .map{ sentence =>

      val result = SENTENCE_START + sentence.result

      val resultWithSentenceEnd = REGEX_SENTENCE_END.findFirstIn(result) match {
        case None => result
        case Some(_) => result.replaceAll("\\R", "") + SENTENCE_END
      }

      sentence.copy(

        result = resultWithSentenceEnd
      )
    }
  }
}
