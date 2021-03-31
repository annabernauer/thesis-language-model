package com.mailytica.thesis.language.model.evaluation.annotators.ngram

import breeze.numerics.{log, sqrt}
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel}
import com.johnsnowlabs.nlp.AnnotatorType.TOKEN
import com.johnsnowlabs.nlp.annotator.NGramGenerator
import com.mailytica.thesis.language.model.evaluation.annotators.ngram.CustomAnnotationTypes.LANGUAGE_MODEL_ANNOTATION
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.Identifiable
import com.mailytica.thesis.language.model.util.Utility.DELIMITER

class NGramSentenceEvaluationModel(override val uid: String) extends AnnotatorModel[NGramSentenceEvaluationModel] {

  def this() = this(Identifiable.randomUID("NGRAM_SENTENCES"))

  override val inputAnnotatorTypes: Array[String] = Array(TOKEN)

  override val outputAnnotatorType: AnnotatorType = LANGUAGE_MODEL_ANNOTATION

  val n: Param[Int] = new Param(this, "n", "")

  val nGramEvaluationModel: Param[NGramEvaluationModel] = new Param(this, "nGramAnnotatorModel", "")

  def setN(value: Int): this.type = set(this.n, value)

  def setNGramEvaluationModel(value: NGramEvaluationModel): this.type = set(this.nGramEvaluationModel, value)

  setDefault(this.n, 3)

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    val startTime = System.nanoTime
    val nGrams: Seq[Annotation] = getTransformedNGramString(annotations, $(n))
    val nGramsWithProbability: Seq[Annotation] = nGrams.flatMap(annotation => $(nGramEvaluationModel).annotate(Seq(annotation)))
    val duration = (System.nanoTime - startTime) / 1e9d
    val likelihoods: Seq[Double] =
      nGramsWithProbability
        .map(annotation => annotation.metadata.getOrElse("probability", "0.0").toDouble)

    val invertedLikelihoods: Seq[Double] = likelihoods.map(likelihood => 1 / likelihood)
    val perplexity: Double = sqrt(invertedLikelihoods.product)

    val avgLogLikelihood: Double =
      likelihoods
        .map(likelihood => breeze.numerics.log(likelihood))
        .sum / likelihoods.size

    val median = medianCalculator(likelihoods)

    val avgLikelihood: Double = likelihoods.sum / likelihoods.size

    Seq(new Annotation(
      LANGUAGE_MODEL_ANNOTATION,
      0,
      0,
      nGrams.headOption.map(_.result).getOrElse("empty").replace(DELIMITER, " "),
      Map("perplexity" -> perplexity.toString,
        "avgLogLikelihood" -> avgLogLikelihood.toString,
        "avgLikelihood" -> avgLikelihood.toString,
        "median" -> median.toString,
        "duration" -> duration.toString)
    ))
  }

  def medianCalculator(seq: Seq[Double]): Double = {
    val sortedSeq: Seq[Double] = seq.sortWith(_ < _)
    if (seq.size % 2 == 1) sortedSeq(sortedSeq.size / 2)
    else {
      val (up: Seq[Double], down: Seq[Double]) = sortedSeq.splitAt(seq.size / 2)
      (up.lastOption.getOrElse(0.0) + down.headOption.getOrElse(0.0)) / 2
    }
  }


  def getTransformedNGramString(tokens: Seq[Annotation], n: Int): Seq[Annotation] = {

    val nGramModel = new NGramGenerator()
      .setInputCols("tokens")
      .setOutputCol(s"$n" + "ngrams")
      .setN(n)
      .setEnableCumulative(false)
      .setDelimiter(DELIMITER)

    nGramModel.annotate(tokens)
  }

}
