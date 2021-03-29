package com.mailytica.thesis.language.model.evaluation.annotators.ngram

import com.johnsnowlabs.nlp.{Annotation, AnnotatorApproach}
import com.johnsnowlabs.nlp.AnnotatorType.TOKEN
import com.johnsnowlabs.nlp.annotator.NGramGenerator
import com.mailytica.thesis.language.model.util.Utility.DELIMITER
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.Dataset

class NGramEvaluation (override val uid: String) extends AnnotatorApproach[NGramEvaluationModel]{

  override val description: String = "NGramEvaluation"

  override val inputAnnotatorTypes: Array[String] = Array(TOKEN)
  override val outputAnnotatorType: AnnotatorType = TOKEN

  val n: Param[Int] = new Param(this, "n", "")

  def setN(value: Int): this.type = set(this.n, value)

  setDefault(this.n, 3)

  def this() = this(Identifiable.randomUID("NGRAM_ANNOTATOR"))


  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): NGramEvaluationModel = {
    import dataset.sparkSession.implicits._

    val tokensPerDocuments: Seq[Array[Annotation]] = dataset
      .select(getInputCols.head)
      .as[Array[Annotation]]
      .collect()
      .toSeq

    val dictionary: Set[String] = tokensPerDocuments
      .flatten
      .map(token => token.result)
      .toSet

    val histories: Seq[Annotation] = getTransformedNGramString(tokensPerDocuments, $(n) - 1)
    val sequences: Seq[Annotation] = getTransformedNGramString(tokensPerDocuments, $(n))

    val historiesMap: Map[String, Int] = getCountedMap(histories)
    val sequencesMap: Map[String, Int] = getCountedMap(sequences)

    new NGramEvaluationModel()
      .setHistories(historiesMap)
      .setSequences(sequencesMap)
      .setN($(n))
      .setDictionary(dictionary)
  }

  def getCountedMap(sequence: Seq[Annotation]) = {
    sequence
      .groupBy[String](annotation => annotation.result)
      .map { case (key, values) => (key, values.size) }
  }

  def getTransformedNGramString(tokensPerDocuments: Seq[Array[Annotation]], n: Int): Seq[Annotation] = {

    val nGramModel = new NGramGenerator()
      .setInputCols("tokens")
      .setOutputCol(s"$n" + "ngrams")
      .setN(n)
      .setEnableCumulative(false)
      .setDelimiter(DELIMITER)

    tokensPerDocuments.flatMap { tokens =>
      nGramModel.annotate(tokens)
    }
  }

}
