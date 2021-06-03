package com.mailytica.thesis.language.model.ngram.annotators.ngram

import com.johnsnowlabs.nlp.AnnotatorType.{CHUNK, TOKEN}
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel}
import com.mailytica.thesis.language.model.ngram.Timer.nGramGeneratorTimer
import com.mailytica.thesis.language.model.util.Utility.DELIMITER
import org.apache.commons.lang.time.StopWatch
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.Identifiable

import java.util.concurrent.TimeUnit

class NGramCustomGenerator(override val uid: String) extends AnnotatorModel[NGramCustomGenerator] {

  override val outputAnnotatorType: AnnotatorType = CHUNK

  override val inputAnnotatorTypes: Array[String] = Array(TOKEN)

  def this() = this(Identifiable.randomUID("N_GRAM_CUSTOM_GENERATOR"))

  val n: Param[Int] = new Param(this, "n", "")

  val delimiter: Param[String] = new Param[String](this, "delimiter", "Glue character used to join the tokens")

  val nGramMinimum: Param[Int] = new Param(this, "nGramMinimum", "")

  def setNGramMinimum(value: Int): this.type = set(this.nGramMinimum, value)

  def setN(value: Int): this.type = set(this.n, value)

  def setDelimiter(value: String): this.type = {
    set(delimiter, value)
  }

  setDefault(this.n -> 3,
    this.delimiter -> DELIMITER,
    this.nGramMinimum -> 1)

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {

    val stopwatch = new StopWatch
    stopwatch.reset()
    stopwatch.start()

    case class NgramChunkAnnotation(currentChunkIdx: Int, annotations: Seq[Annotation])

    val range = Range.inclusive($(nGramMinimum), $(n))

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

    nGramGeneratorTimer.update(stopwatch.getTime, TimeUnit.MILLISECONDS)

    ngramsAnnotation.annotations

  }
}
