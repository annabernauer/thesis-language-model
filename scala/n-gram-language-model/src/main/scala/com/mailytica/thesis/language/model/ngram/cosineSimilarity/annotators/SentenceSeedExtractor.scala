package com.mailytica.thesis.language.model.ngram.cosineSimilarity.annotators

import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, AnnotatorType}
import com.johnsnowlabs.nlp.AnnotatorType.{CHUNK, DOCUMENT, TOKEN}
import com.mailytica.thesis.language.model.ngram.annotators.RedundantTextTrimmer
import com.mailytica.thesis.language.model.util.Utility.DELIMITER
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}

class SentenceSeedExtractor(override val uid: String) extends AnnotatorModel[SentenceSeedExtractor] with DefaultParamsWritable {

  override val inputAnnotatorTypes: Array[String] = Array(TOKEN)

  override val outputAnnotatorType: AnnotatorType = TOKEN

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

    val bound = annotations.length match {
      case k if (k < $(n)) => k
      case _ => $(n)
    }
    //    val range: Seq[Annotation] = Range.inclusive(1, $(n)).map(i => annotations(i)).foldLeft(Seq.empty[Annotation]){(annotations : Seq[Annotation], i) => annotations :+ i}
    //    val range: Seq[Annotation] = Range.inclusive(1, $(n)).foldLeft(Seq.empty[Annotation]){(annotationsChunk : Seq[Annotation], i) => annotationsChunk :+ annotations(i)}
    val sentenceSeed: Seq[String] = Range.inclusive(0, bound-1).foldLeft(Seq.empty[String]) { (annotationsChunk: Seq[String], i) => annotationsChunk :+ annotations(i).result }

    Seq(Annotation(
      annotatorType = AnnotatorType.DOCUMENT,
      result = sentenceSeed.mkString(" "),
      begin = 0,
      end = 0,
      metadata = Map.empty
    ))
  }
}

object SentenceSeedExtractor extends DefaultParamsReadable[SentenceSeedExtractor] {

}
