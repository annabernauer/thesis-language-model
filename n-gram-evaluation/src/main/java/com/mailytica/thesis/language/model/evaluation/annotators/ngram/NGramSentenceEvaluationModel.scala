package com.mailytica.thesis.language.model.evaluation.annotators.ngram

import breeze.numerics.{log, sqrt}
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel}
import com.johnsnowlabs.nlp.AnnotatorType.TOKEN
import com.johnsnowlabs.nlp.annotator.NGramGenerator
import com.johnsnowlabs.nlp.serialization.{MapFeature, SetFeature}
import com.mailytica.thesis.language.model.evaluation.annotators.ngram.CustomAnnotationTypes.LANGUAGE_MODEL_ANNOTATION
import org.apache.spark.ml.param.{IntArrayParam, IntParam, Param, StringArrayParam}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import com.mailytica.thesis.language.model.util.Utility.DELIMITER

class NGramSentenceEvaluationModel(override val uid: String) extends AnnotatorModel[NGramSentenceEvaluationModel] with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("NGRAM_SENTENCES_EVALUATION"))

  override val inputAnnotatorTypes: Array[String] = Array(TOKEN)

  override val outputAnnotatorType: AnnotatorType = LANGUAGE_MODEL_ANNOTATION

  val n: IntParam = new IntParam(this, "n", "")

  val historyKeys: StringArrayParam = new StringArrayParam(this, "historyKeys", "")
  val historyValues: IntArrayParam = new IntArrayParam(this, "historyValues", "")

  val sequenceKeys: StringArrayParam = new StringArrayParam(this, "sequenceKeys", "")
  val sequenceValues: IntArrayParam = new IntArrayParam(this, "sequenceValues", "")

  val dictionaryArray: StringArrayParam = new StringArrayParam(this, "dictionary", "")

  def setHistoryKeys(value: Array[String]): this.type = set(historyKeys, value)

  def setHistoryValues(value: Array[Int]): this.type = set(historyValues, value)

  def setSequenceKeys(value: Array[String]): this.type = set(sequenceKeys, value)

  def setSequenceValues(value: Array[Int]): this.type = set(sequenceValues, value)

  def setDictionary(value: Array[String]): this.type = set(dictionaryArray, value)

  def setN(value: Int): this.type = set(this.n, value)

  setDefault(this.n, 3)

  lazy val nGramEvaluationModel = new NGramEvaluationModel()
    .setSequences($(sequenceKeys).zip($(sequenceValues)).toMap)
    .setHistories($(historyKeys).zip($(historyValues)).toMap)
    .setDictionary($(dictionaryArray).toSet)
    .setN($(n))

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {

    val startTime = System.nanoTime
    val nGrams: Seq[Annotation] = getTransformedNGramString(annotations, $(n))
    val nGramsWithProbability: Seq[Annotation] = nGrams.flatMap(annotation => nGramEvaluationModel.annotate(Seq(annotation)))
    val duration = (System.nanoTime - startTime) / 1e9d
    val likelihoods: Seq[Double] =
      nGramsWithProbability
        .map(annotation => annotation.metadata.getOrElse("probability", "0.0").toDouble)


    val invertedLikelihoods: Seq[Double] = likelihoods.map(likelihood => 1 / likelihood)

    //    invertedLikelihoods.foreach(println)

    val perplexity: Double = invertedLikelihoods.map(lh => Math.pow(lh, 1 / invertedLikelihoods.size.toDouble)).product

    if (perplexity.isInfinite) {
      //      invertedLikelihoods.foreach(println)
      //      likelihoods.foreach(println)

      //      val likelihoods2: Seq[(Double, String)] =
//        nGramsWithProbability
//          .map(annotation => {
//            val likelih = annotation.metadata.getOrElse("probability", "0.0").toDouble
//            (1 / likelih, annotation.result)
//          })
//      likelihoods2.foreach(println)
      println("WARNING perplexity is infinite")
    }

    //    likelihoods2.foreach(a => println(a._1 + " " + a._2 + " " + a._3))
    //    invertedLikelihoods.foreach(println)
    //    println("produkt " + invertedLikelihoods.product)
    //    println("size " + 1 / invertedLikelihoods.size.toDouble)

    //    println()

    val avgLogLikelihood: Double =
      likelihoods
        .map(likelihood => breeze.numerics.log(likelihood))
        .sum / likelihoods.size

    val medianLikelihoods = medianCalculator(likelihoods)

    val avgLikelihood: Double = likelihoods.sum / likelihoods.size

    Seq(new Annotation(
      LANGUAGE_MODEL_ANNOTATION,
      0,
      0,
      nGrams.headOption.map(_.result).getOrElse("empty").replace(DELIMITER, " "),
      Map("perplexity" -> perplexity.toString,
        "avgLogLikelihood" -> avgLogLikelihood.toString,
        "avgLikelihood" -> avgLikelihood.toString,
        "medianLikelihoods" -> medianLikelihoods.toString,
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

    val nGramModel = new NGramCustomGenerator()
      .setInputCols("tokens")
      .setOutputCol(s"$n" + "ngrams")
      .setN(n)
      .setNGramMinimum(2) //generated ngrams aren't allowed to have a length n < 2 (has to be n - 1 > 0 )
      //      .setEnableCumulative(false)
      .setDelimiter(DELIMITER)

    nGramModel.annotate(tokens)
  }

}

object NGramSentenceEvaluationModel extends DefaultParamsReadable[NGramSentenceEvaluationModel] {

}
