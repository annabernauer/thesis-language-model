package com.mailytica.thesis.language.model.ngram.annotators

import com.johnsnowlabs.nlp.AnnotatorType.DOCUMENT
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, AnnotatorType}
import org.apache.spark.ml.util.Identifiable

import scala.util.matching.Regex

class SentenceSplitter(override val uid: String) extends AnnotatorModel[SentenceSplitter] {

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
          .foldLeft(Seq.empty[Annotation]) { case (sentences: Seq[Annotation], sentenceBoundaryMatch: Regex.Match) => //first param is acc and is initialized with a start value
                                                                                                                      //second param is the list/seq... iterated
            sentences :+ Annotation(
              annotatorType = AnnotatorType.DOCUMENT,
              result = result.substring(sentences.lastOption.map(_.`end`).getOrElse(0), sentenceBoundaryMatch.end),
              begin = sentences.lastOption.map(_.`end`).getOrElse(0),
              end = sentenceBoundaryMatch.end,
              metadata = Map("sentence" -> sentences.length.toString)
            )
          }.filterNot(sentence => sentence.result.trim.isEmpty)

        sentences
      }

  }
}
