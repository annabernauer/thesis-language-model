package com.mailytica.thesis.language.model.ngram.cosineSimilarity.annotators

import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel}
import com.johnsnowlabs.nlp.AnnotatorType.DOCUMENT
import org.apache.spark.ml.util.Identifiable

import scala.util.matching.Regex

class SentenceNewLineRemover(override val uid: String) extends AnnotatorModel[SentenceNewLineRemover] {

  override val outputAnnotatorType: AnnotatorType = DOCUMENT

  override val inputAnnotatorTypes: Array[String] = Array(DOCUMENT)

  def this() = this(Identifiable.randomUID("SENTENCE_END_MARKER"))

  val SENTENCE_END: String = " <SENTENCE_END>"

  val SENTENCE_START: String = "<SENTENCE_START> "

  val REGEX_SENTENCE_END: Regex = "(\\.|\\!|\\?|\\:|\\R)$".r

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {

    annotations
      .map { sentence =>

        val result = sentence.result

        val resultWithSentenceEnd = REGEX_SENTENCE_END.findFirstIn(result) match {
          case None => result
          case Some(_) => result.replaceAll("\\R", "")
        }

        sentence.copy(
          result = resultWithSentenceEnd
        )
      }
    //      .filterNot(sentence => sentence.result.startsWith(SENTENCE_END))
  }

}
