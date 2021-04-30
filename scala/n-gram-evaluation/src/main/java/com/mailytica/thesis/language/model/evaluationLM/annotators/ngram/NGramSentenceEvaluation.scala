package com.mailytica.thesis.language.model.evaluationLM.annotators.ngram

import com.johnsnowlabs.nlp.AnnotatorApproach
import com.johnsnowlabs.nlp.AnnotatorType.TOKEN
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.Dataset

class NGramSentenceEvaluation (override val uid: String) extends AnnotatorApproach[NGramSentenceEvaluationModel] {

  override val description: String = "NGRAM_SENTENCE_EVALUATION"

  override val inputAnnotatorTypes: Array[String] = Array(TOKEN)
  override val outputAnnotatorType: AnnotatorType = TOKEN

  val n: Param[Int] = new Param(this, "n", "")

  def setN(value: Int): this.type = set(this.n, value)

  setDefault(this.n, 3)

  def this() = this(Identifiable.randomUID("NGRAM_SENTENCES"))

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): NGramSentenceEvaluationModel = {

    val nGramEvaluation = new NGramEvaluation()
      .setInputCols("token")
      .setOutputCol("ngramsProbability")
      .setN($(n))

    val model : NGramEvaluationModel = nGramEvaluation.train(dataset)


    val sequencesArray : Array[(String, Int)] = model.getSequences.toArray             //important for having the same index in SequenceKeys and SequenceValues
    val historiesArray : Array[(String, Int)] = model.getHistories.toArray             //important for having the same index in historyKeys and historyValues

    new NGramSentenceEvaluationModel()
      .setSequenceKeys(sequencesArray.map(_._1))
      .setSequenceValues(sequencesArray.map(_._2))
      .setHistoryKeys(historiesArray.map(_._1))
      .setHistoryValues(historiesArray.map(_._2))
      .setDictionary(model.getDictionary.toArray)

  }

}
