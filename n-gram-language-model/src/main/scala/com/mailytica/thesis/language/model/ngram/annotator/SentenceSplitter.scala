package com.mailytica.thesis.language.model.ngram.annotator

import com.johnsnowlabs.nlp.AnnotatorType.DOCUMENT
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, AnnotatorType}
import org.apache.spark.ml.util.Identifiable

import scala.util.matching.Regex

class SentenceSplitter(override val uid: String) extends AnnotatorModel[SentenceSplitter] {

  override val outputAnnotatorType: AnnotatorType = DOCUMENT

  override val inputAnnotatorTypes: Array[String] = Array(DOCUMENT)

  def this() = this(Identifiable.randomUID("SENTENCE_SPLITTER"))

  val SENTENCE_END: String = " <SENTENCE_END>"

  val REGEX_SENTENCE_END: String = "(\\.|\\!|\\?|\\:|\\R)$"

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {


    annotations
      .map { document =>

        val result = document.result

        val sentences: Seq[Annotation] = "(\\.|:|\\R|\\?|\\!|$)"
          .r()
          .findAllMatchIn(result)
          .foldLeft(Seq.empty[Annotation]) { case (sentences : Seq[Annotation], sentenceBoundaryMatch : Regex.Match) =>
            sentences :+ Annotation(
              annotatorType = AnnotatorType.DOCUMENT,
              result = result.substring(sentences.lastOption.map(_.`end`).getOrElse(0), sentenceBoundaryMatch.end),
              begin = sentences.lastOption.map(_.`end`).getOrElse(0),
              end = sentenceBoundaryMatch.end,
              metadata = Map("sentence" -> sentences.length.toString)
            )
          }

        sentences
      }
    val changedAnnotations: Seq[Annotation] =
      annotations
        .flatMap(document =>
          document
            .result
            .split(REGEX_SENTENCE_END)
            .zipWithIndex
            .map {
              case (result, index) => document.copy(result = result, metadata = Map(("sentence", index.toString)))
            })

    changedAnnotations
  }

}
