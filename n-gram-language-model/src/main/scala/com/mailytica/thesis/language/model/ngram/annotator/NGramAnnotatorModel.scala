package com.mailytica.thesis.language.model.ngram.annotator

import com.johnsnowlabs.nlp.AnnotatorType.{CHUNK, TOKEN}
import com.johnsnowlabs.nlp.annotator.NGramGenerator
import com.johnsnowlabs.nlp.serialization.MapFeature
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel}
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.Identifiable

import scala.util.Try

class NGramAnnotatorModel(override val uid: String) extends AnnotatorModel[NGramAnnotatorModel] {

  def this() = this(Identifiable.randomUID("NGRAM"))

  val histories: MapFeature[String, Int] = new MapFeature(this, "histories")
  val sequences: MapFeature[String, Int] = new MapFeature(this, "sequences")
  val n: Param[Int] = new Param(this, "n", "")

  def setN(value: Int): this.type = set(this.n, value)

  def setHistories(value: Map[String, Int]): this.type = set(histories, value)

  def setSequences(value: Map[String, Int]): this.type = set(sequences, value)

  setDefault(this.n, 3)

  override val inputAnnotatorTypes: Array[String] = Array(TOKEN)
  override val outputAnnotatorType: AnnotatorType = CHUNK

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {

    val dictionary: Set[String] = Set("million", "Quantum", "test")

    val nGrams: Seq[Annotation] = getTransformedNGramString(annotations, $(n) - 1)

    def calculateTokenWithLMaxLikelihood(ngram: Annotation): Option[String] = {

      def getTokenWithLikelihood(token: String): (String, Double) = {

        val likelihood: Double = Try {
          $$(sequences).getOrElse[Int](s"${ngram.result} $token", 0).toDouble / $$(histories).getOrElse[Int](ngram.result, 0).toDouble
        }.getOrElse(0.0)

        (token, likelihood)
      }

      dictionary
        .toSeq
        .map { token => getTokenWithLikelihood(token) }
        .sortBy { case (token, likelihood) => likelihood }
        .lastOption
        .map { case (token, likelihood) => token }

    }

    val tokenWithMaxLikelihood: String = nGrams
      .lastOption
      .flatMap(ngram => calculateTokenWithLMaxLikelihood(ngram))
      .getOrElse("noStringFound")

    Seq(Annotation(TOKEN, 0, tokenWithMaxLikelihood.length, tokenWithMaxLikelihood, Map((""->"")), Array(1F)))
  }


  def getTransformedNGramString(tokens: Seq[Annotation], n: Int): Seq[Annotation] = {

    val nGramModel = new NGramGenerator()
      .setInputCols("tokens")
      .setOutputCol(s"$n" + "ngrams")
      .setN(n)
      .setEnableCumulative(false)

    nGramModel.annotate(tokens)
  }

}
