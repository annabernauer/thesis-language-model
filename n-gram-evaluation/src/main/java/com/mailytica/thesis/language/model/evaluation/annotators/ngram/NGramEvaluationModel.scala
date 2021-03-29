package com.mailytica.thesis.language.model.evaluation.annotators.ngram

import com.johnsnowlabs.nlp.AnnotatorType.TOKEN
import com.johnsnowlabs.nlp.serialization.{MapFeature, SetFeature}
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel}
import com.mailytica.thesis.language.model.util.Utility.DELIMITER
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.Identifiable

import scala.util.Try

class NGramEvaluationModel(override val uid: String) extends AnnotatorModel[NGramEvaluationModel] {

  override val inputAnnotatorTypes: Array[String] = Array(TOKEN)
  override val outputAnnotatorType: AnnotatorType = TOKEN

  val histories: MapFeature[String, Int] = new MapFeature(this, "histories")

  val sequences: MapFeature[String, Int] = new MapFeature(this, "sequences")

  val dictionary: SetFeature[String] = new SetFeature(this, "dictionary")

  val n: Param[Int] = new Param(this, "n", "")


  def setHistories(value: Map[String, Int]): this.type = set(histories, value)

  def setSequences(value: Map[String, Int]): this.type = set(sequences, value)

  def setDictionary(value: Set[String]): this.type = set(dictionary, value)

  def setN(value: Int): this.type = set(this.n, value)

  setDefault(this.n, 3)

  def this() = this(Identifiable.randomUID("NGRAM_EVALUATION_MODEL"))

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {

    def getLikelihood(ngram: String): Double = {

      val historyString =
        ngram
          .split("\\" + DELIMITER)
          .dropRight(1)
          .mkString(DELIMITER)

      val likelihood: Double = Try {
        $$(sequences).getOrElse[Int](ngram, 0).toDouble / $$(histories).getOrElse[Int](historyString, 0).toDouble
      }.getOrElse(0.0)

      likelihood
    }

    val likelihood: Double = annotations
      .lastOption
      .map(ngram => getLikelihood(ngram.result))
      .getOrElse(0.0)


    annotations
      .lastOption
      .map(annotation => annotation.copy(metadata = annotation.metadata + ("probability" -> likelihood.toString))).toSeq
  }

}
