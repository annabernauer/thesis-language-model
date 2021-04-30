package com.mailytica.thesis.language.model.evaluationLM.annotators.ngram

import com.johnsnowlabs.nlp.AnnotatorType.TOKEN
import com.johnsnowlabs.nlp.serialization.{MapFeature, SetFeature}
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel}
import com.mailytica.thesis.language.model.util.Utility.DELIMITER
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}

import scala.collection.immutable
import scala.runtime.Tuple2Zipped
import scala.util.Try

class NGramEvaluationModel(override val uid: String) extends AnnotatorModel[NGramEvaluationModel] with DefaultParamsWritable {

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

  def getN: Int = $(n)

  def getHistories: Map[String, Int] = $$(histories)

  def getSequences: Map[String, Int] = $$(sequences)

  def getDictionary: Set[String] = $$(dictionary)

  setDefault(this.n, 3)

  def this() = this(Identifiable.randomUID("NGRAM_EVALUATION_MODEL"))

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {

    def getLikelihood(ngram: String): Double = {

      val allNgrams: Seq[String] = getAllNgrams(Seq(ngram)) //for smoothing

      val likelihood: Double = allNgrams.length match {
        case 1 => 0.0 //n is to small, has to be n > 1 (because n - 1 > 0 )
        case _ => Try {
          val nGramProbabilities: List[Double] =
            List.range(0, allNgrams.length - 1)
              .map((index) =>
                $$(sequences).getOrElse[Int](allNgrams(index), 0).toDouble / $$(histories).getOrElse[Int](allNgrams(index + 1), 0).toDouble)

          val nGramProbabilitiesWithoutInfinite: Seq[Double] = nGramProbabilities.map {
            case v if v.isInfinite => 0.0 //is infinite when divided by zero
            case v => v }

          val soonToBeMultiplicators: Seq[Double] = Range.inclusive(2, $(n)).map(index => index / nGramProbabilities.length.toDouble)
          val sumMultiplicators = soonToBeMultiplicators.sum
          val finalMultiplicators = soonToBeMultiplicators.map(multiplicator => multiplicator / sumMultiplicators)                                            //(n/N)/SUM(n/N)

          val probability: Double = (nGramProbabilitiesWithoutInfinite, finalMultiplicators).zipped.map {case (nGramProbability, multiplicator) => nGramProbability * multiplicator }.sum

          probability
        }.getOrElse(0.0)
      }

      if (likelihood.isInfinite || likelihood.isNaN) {
        0.0
      } else {
        likelihood
      }
    }

    val likelihood: Double = annotations
      .lastOption
      .map(ngram => getLikelihood(ngram.result))
      .getOrElse(0.0)

    annotations
      .lastOption
      .map(annotation => annotation.copy(metadata = annotation.metadata + ("probability" -> likelihood.toString))).toSeq
  }

  def getAllNgrams(ngramSeq: Seq[String]): Seq[String] = {
    val splittedNgram = ngramSeq.last
      .split(DELIMITER)

    if (splittedNgram.length <= 1) {
      return ngramSeq
    }

    val nMinusOneGram =
      splittedNgram
        .dropRight(1)
        .mkString(DELIMITER)

    getAllNgrams(ngramSeq ++ Seq(nMinusOneGram))
  }
}

object NGramEvaluationModel extends DefaultParamsReadable[NGramEvaluationModel] {

}
