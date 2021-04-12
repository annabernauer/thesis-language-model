package com.mailytica.thesis.language.model.evaluation.annotators

import com.johnsnowlabs.nlp.AnnotatorType.DOCUMENT
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, AnnotatorType}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}

import scala.util.matching.Regex

class SentenceSplitter(override val uid: String) extends AnnotatorModel[SentenceSplitter] with DefaultParamsWritable {

  override val outputAnnotatorType: AnnotatorType = DOCUMENT

  override val inputAnnotatorTypes: Array[String] = Array(DOCUMENT)

  def this() = this(Identifiable.randomUID("SENTENCE_SPLITTER"))

  val SENTENCE_END: String = " <SENTENCE_END>"

  val REGEX_SENTENCE_END: Regex = "(\\.|:|\\R|\\?|\\!|$)".r()

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {

    annotations
      .flatMap { document =>

        val result = document.result

        val sentences: Seq[Annotation] = REGEX_SENTENCE_END
          .findAllMatchIn(result)
          .foldLeft(Seq.empty[Annotation]) { case (sentences: Seq[Annotation], sentenceBoundaryMatch: Regex.Match) =>
            sentences :+ Annotation(
              annotatorType = AnnotatorType.DOCUMENT,
              result = result.substring(sentences.lastOption.map(_.`end`).getOrElse(0), sentenceBoundaryMatch.end),
              begin = sentences.lastOption.map(_.`end`).getOrElse(0),
              end = sentenceBoundaryMatch.end,
              metadata = Map("sentence" -> sentences.length.toString)
            )
          }.filterNot(sentence => sentence.result.trim.isEmpty)

        sentences
          .map{ sentence =>

            val result = sentence.result

            val resultWithSentenceEnd = REGEX_SENTENCE_END.findFirstIn(result) match {            // essential for sentence prediction, because otherwise not finished sentences are getting terminated and sentence completion isn't possible
              case None => result
              case Some(_) => result.replaceAll("\\R", "") + SENTENCE_END
            }

            sentence.copy(
              result = resultWithSentenceEnd
            )
          }

        sentences
      }

  }
}

object SentenceSplitter extends DefaultParamsReadable[SentenceSplitter] {

}
