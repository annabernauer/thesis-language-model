package com.mailytica.thesis.language.model.ngram.cosineSimilarity.annotators

import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, AnnotatorType}
import com.johnsnowlabs.nlp.AnnotatorType.{CHUNK, DOCUMENT, TOKEN}
import com.mailytica.thesis.language.model.ngram.annotators.RedundantTextTrimmer
import com.mailytica.thesis.language.model.util.Utility.DELIMITER
import org.apache.commons.cli.Option
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}

import scala.Option

class SentenceSeedExtractor(override val uid: String) extends AnnotatorModel[SentenceSeedExtractor] with DefaultParamsWritable {

  override val inputAnnotatorTypes: Array[String] = Array(TOKEN)

  override val outputAnnotatorType: AnnotatorType = CHUNK

  val SENTENCE_END: String = "<SENTENCE_END>"

  def this() = this(Identifiable.randomUID("SENTENCE_SEED_EXTRACTOR"))

  val n: Param[Int] = new Param(this, "n", "")

  val delimiter: Param[String] = new Param[String](this, "delimiter", "Glue character used to join the tokens")

  def setN(value: Int): this.type = set(this.n, value)

  def setDelimiter(value: String): this.type = {
    set(delimiter, value)
  }

  setDefault(this.n -> 3,
    this.delimiter -> DELIMITER)

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {

    val seedLength = $(n) - 1
    annotations.length match {
      case length if (length < seedLength) => //seed would be smaller than n, needs to be eliminated
        Seq.empty
      case _ =>
        val sentenceSeed: Seq[String] = Range.inclusive(0, seedLength - 1).foldLeft(Seq.empty[String]) { (annotationsChunk: Seq[String], i) => annotationsChunk :+ annotations(i).result }
        sentenceSeed.lastOption.getOrElse("") match {
          case SENTENCE_END => //sentence is already finished with End Tag
            Seq.empty
          case _ =>
            Seq(Annotation(
              annotatorType = AnnotatorType.DOCUMENT,
              result = sentenceSeed.mkString(" "),
              begin = 0,
              end = 0,
              metadata = Map.empty
            ))
        }

    }
  }
}

object SentenceSeedExtractor extends DefaultParamsReadable[SentenceSeedExtractor] {

}
