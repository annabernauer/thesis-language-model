package com.mailytica.thesis.language.model.ngram.annotator

import com.johnsnowlabs.nlp.AnnotatorType.DOCUMENT
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel}
import com.johnsnowlabs.nlp.annotators.NGramGenerator
import org.apache.spark.ml.util.Identifiable

import scala.util.matching.Regex

class SentenceEndMarker(override val uid: String) extends AnnotatorModel[SentenceEndMarker] {

  override val outputAnnotatorType: AnnotatorType = DOCUMENT

  override val inputAnnotatorTypes: Array[String] = Array(DOCUMENT)

  def this() = this(Identifiable.randomUID("SENTENCE_END_MARKER"))

  val SENTENCE_END: String = " <SENTENCE_END>"

  val REGEX_SENTENCE_END: Regex = "(\\.|\\!|\\?|\\:|\\R)$".r

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {

    annotations.map{ sentence =>

      val result = sentence.result

      val resultWithSentenceEnd = REGEX_SENTENCE_END.findFirstIn(result) match {
        case None => result
        case Some(_) => result + SENTENCE_END
      }

      println("resultWithSentenceEnd")
      println(resultWithSentenceEnd)
      sentence.copy(

        result = resultWithSentenceEnd
      )
    }
  }

}
