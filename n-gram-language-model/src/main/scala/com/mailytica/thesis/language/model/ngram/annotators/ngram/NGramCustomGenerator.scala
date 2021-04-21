package com.mailytica.thesis.language.model.ngram.annotators.ngram

import com.johnsnowlabs.nlp.AnnotatorType.{CHUNK, TOKEN}
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel}
import com.mailytica.thesis.language.model.util.Utility.DELIMITER
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.Identifiable

class NGramCustomGenerator(override val uid: String) extends AnnotatorModel[NGramCustomGenerator] {

  override val outputAnnotatorType: AnnotatorType = TOKEN

  override val inputAnnotatorTypes: Array[String] = Array(CHUNK)

  def this() = this(Identifiable.randomUID("N_GRAM_CUSTOM_GENERATOR"))

  val n: Param[Int] = new Param(this, "n", "")

  val delimiter: Param[String] = new Param[String](this, "delimiter", "Glue character used to join the tokens")

  def setN(value: Int): this.type = set(this.n, value)

  def setDelimiter(value: String): this.type = {
    set(delimiter, value)
  }

  setDefault(this.n -> 3,
    this.delimiter -> DELIMITER)

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {

    case class NgramChunkAnnotation(currentChunkIdx: Int, annotations: Seq[Annotation])

    val range = Range.inclusive(1, $(n))

    val ngramsAnnotation = range.foldLeft(NgramChunkAnnotation(0, Seq[Annotation]()))((currentNgChunk, k) => {

      val chunksForCurrentWindow = annotations.iterator.sliding(k).withPartial(false).zipWithIndex.map { case (tokens: Seq[Annotation], localChunkIdx: Int) =>
        Annotation(
          outputAnnotatorType,
          tokens.head.begin,
          tokens.last.end,
          tokens.map(_.result).mkString($(delimiter)),
          Map(
            "sentence" -> tokens.head.metadata.getOrElse("sentence", "0"),
            "chunk" -> tokens.head.metadata.getOrElse("chunk", (currentNgChunk.currentChunkIdx + localChunkIdx).toString)
          )
        )
      }.toArray
      NgramChunkAnnotation(currentNgChunk.currentChunkIdx + chunksForCurrentWindow.length, currentNgChunk.annotations ++ chunksForCurrentWindow)
    })
    ngramsAnnotation.annotations

  }
}
