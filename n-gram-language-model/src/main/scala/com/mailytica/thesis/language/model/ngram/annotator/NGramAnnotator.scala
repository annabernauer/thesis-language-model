package com.mailytica.thesis.language.model.ngram.annotator

import com.johnsnowlabs.nlp.AnnotatorType.{CHUNK, DOCUMENT, TOKEN}
import com.johnsnowlabs.nlp.annotator.{NGramGenerator, Tokenizer}
import com.johnsnowlabs.nlp.serialization.MapFeature
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.nlp.{Annotation, AnnotatorApproach, DocumentAssembler, LightPipeline}
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.Dataset

class NGramAnnotator(override val uid: String) extends AnnotatorApproach[NGramAnnotatorModel] {


  override val inputAnnotatorTypes: Array[String] = Array(TOKEN)
  override val outputAnnotatorType: AnnotatorType = CHUNK

  val n: Param[Int] = new Param(this, "n", "")

  def setN(value: Int): this.type = set(this.n, value)

  def getN: Int = $(n)

  setDefault(this.n, 3)

  def this() = this(Identifiable.randomUID("NGRAM_ANNOTATOR"))

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): NGramAnnotatorModel = {
    import dataset.sparkSession.implicits._

    val tokensPerDocuments: Seq[Array[Annotation]] = dataset
      .select(getInputCols.head)
      .as[Array[Annotation]]
      .collect()
      .toSeq

    val histories: Seq[Annotation] = getTransformedNGramString(tokensPerDocuments, getN - 1)
    val sequences: Seq[Annotation] = getTransformedNGramString(tokensPerDocuments, getN)

    val historiesMap: Map[String, Int] = getCountedMap(histories)
    val sequencesMap: Map[String, Int] = getCountedMap(sequences)

    new NGramAnnotatorModel()
      .setHistories(historiesMap)
      .setSequences(sequencesMap)
      .setN($(n))
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

    tokensPerDocuments.flatMap { tokens =>
      nGramModel.annotate(tokens)
    }
  }

  override val description: String = "NGrammAnnotator"
}
