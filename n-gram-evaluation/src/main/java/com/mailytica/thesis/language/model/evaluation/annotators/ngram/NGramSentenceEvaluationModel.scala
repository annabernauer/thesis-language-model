package com.mailytica.thesis.language.model.evaluation.annotators.ngram

import com.johnsnowlabs.nlp
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel}
import com.johnsnowlabs.nlp.AnnotatorType.TOKEN
import com.johnsnowlabs.nlp.annotator.NGramGenerator
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.Identifiable

class NGramSentenceEvaluationModel (override val uid: String) extends AnnotatorModel[NGramSentenceEvaluationModel] {

  def this() = this(Identifiable.randomUID("NGRAM_SENTENCES"))

  override val inputAnnotatorTypes: Array[String] = Array(TOKEN)

  override val outputAnnotatorType: AnnotatorType = TOKEN

  val n: Param[Int] = new Param(this, "n", "")

  val nGramEvaluationModel: Param[NGramEvaluationModel] = new Param(this, "nGramAnnotatorModel", "")

  def setN(value: Int): this.type = set(this.n, value)

  def setNGramEvaluationModel(value: NGramEvaluationModel): this.type = set(this.nGramEvaluationModel, value)

  setDefault(this.n, 3)

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {

    val nGrams: Seq[Annotation] = getTransformedNGramString(annotations, $(n))
    val nGramsWithProbability: Seq[nlp.Annotation] = nGrams.flatMap(annotation => $(nGramEvaluationModel).annotate(Seq(annotation)))

    nGramsWithProbability
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
