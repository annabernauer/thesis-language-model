package com.mailytica.thesis.language.model.ngram.annotators.ngram

import com.johnsnowlabs.nlp.AnnotatorType.TOKEN
import com.johnsnowlabs.nlp.annotator.NGramGenerator
import com.johnsnowlabs.nlp.serialization.{MapFeature, SetFeature}
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel}
import com.mailytica.thesis.language.model.util.Utility.DELIMITER
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.Identifiable

import scala.util.Try

class NGramAnnotatorModel(override val uid: String) extends AnnotatorModel[NGramAnnotatorModel] {

  def this() = this(Identifiable.randomUID("NGRAM"))

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


  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {

    def calculateTokenWithLMaxLikelihood(ngram: Annotation): Option[(String, Double)] = {

      def getTokenWithLikelihood(token: String): (String, Double) = {

        val likelihood: Double = Try {
          $$(sequences).getOrElse[Int](s"${ngram.result}$DELIMITER$token", 0).toDouble / $$(histories).getOrElse[Int](ngram.result, 0).toDouble
        }.getOrElse(0.0)
        (token, likelihood)
      }

      val tokensWithLikelihood : Seq[(String, Double)] = $$(dictionary)
        .toSeq // needs to be done for sorting
        .map { token => getTokenWithLikelihood(token) }
        .sortBy { case (token, likelihood) => likelihood }

//      printToFile(new File(s"target\\sentencePrediction\\likelihood_${ngram.result.replace(" ", "_").replaceAll("[,|.|\"]", "sz")}.txt")) { p =>
//        tokensWithLikelihood.foreach(p.println)
//      }

      tokensWithLikelihood
        .lastOption

    }

    val nGrams: Seq[Annotation] = getTransformedNGramString(annotations, $(n) - 1)

    val tokenWithMaxLikelihood: Option[(String, Double)] = nGrams
      .lastOption
      .flatMap(ngram => calculateTokenWithLMaxLikelihood(ngram))

    val (lastTokenEnd: Int, lastTokenSentence: String) =
      annotations.lastOption match {
        case Some(annotation) => (annotation.end + 2, annotation.metadata.getOrElse("sentence", "0"))
        case None => (0, "0")
      }

    tokenWithMaxLikelihood match {
      case Some((tokenContent, tokenProbability)) =>
        Seq(Annotation(
          TOKEN,
          lastTokenEnd,
          lastTokenEnd + tokenContent.length,
          tokenContent,
          Map("probability" -> tokenProbability.toString, "sentence" -> lastTokenSentence))
        )
      case None => Seq.empty
    }
  }


  def getTransformedNGramString(tokens: Seq[Annotation], n: Int): Seq[Annotation] = {

    val nGramModel = new NGramCustomGenerator()
      .setInputCols("tokens")
      .setOutputCol(s"$n" + "ngrams")
      .setN(n)

    nGramModel.annotate(tokens)
  }


  def printToFile(f: java.io.File)(op: java.io.PrintWriter => Unit) {
    val p = new java.io.PrintWriter(f)
    try { op(p) } finally { p.close() }
  }
}
