package com.mailytica.thesis.language.model.ngram.annotators.ngram

import com.johnsnowlabs.nlp.AnnotatorApproach
import com.johnsnowlabs.nlp.AnnotatorType.TOKEN
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.Dataset

class NGramSentenceAnnotator (override val uid: String) extends AnnotatorApproach[NGramSentenceAnnotatorModel]{
  override val description: String = "NGRAM_SENTENCE_ANNOTATOR"

  override val inputAnnotatorTypes: Array[String] = Array(TOKEN)
  override val outputAnnotatorType: AnnotatorType = TOKEN

  val n: Param[Int] = new Param(this, "n", "")

  def setN(value: Int): this.type = set(this.n, value)

  setDefault(this.n, 3)

  def this() = this(Identifiable.randomUID("NGRAM_SENTENCES"))

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): NGramSentenceAnnotatorModel = {

    val nGramAnnotator = new NGramAnnotator()
      .setInputCols("token")
      .setOutputCol("ngramsProbability")
      .setN($(n))

    val model = nGramAnnotator.train(dataset)

    new NGramSentenceAnnotatorModel()
      .setNGramAnnotatorModel(model)

  }


}
