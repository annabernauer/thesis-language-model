package com.mailytica.thesis.language.model.ngram.annotator

import com.johnsnowlabs.nlp.AnnotatorType.{CHUNK, DOCUMENT}
import com.johnsnowlabs.nlp.annotator.{NGramGenerator, Tokenizer}
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.nlp.{Annotation, AnnotatorApproach, DocumentAssembler, LightPipeline}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.Dataset

class NGramAnnotator(override val uid: String) extends AnnotatorApproach[NGramAnnotatorModel] {


  override val inputAnnotatorTypes: Array[String] = Array(DOCUMENT)
  override val outputAnnotatorType: AnnotatorType = CHUNK

  def this() = this(Identifiable.randomUID("NGRAM_ANNOTATOR"))

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): NGramAnnotatorModel = {
    import dataset.sparkSession.implicits._

    val results = dataset
      .select(getInputCols.head)
      .as[Array[Annotation]]
      .flatMap(_.map(_.result))
      .as[String]
      .collect()
      .toSeq

    val histories: Seq[Annotation] = getTransformedNGramString(results, 2)
    val sequences: Seq[Annotation] = getTransformedNGramString(results, 3)

    val historiesMap: Map[String, Int] = getCountedMap(histories)
    val sequencesMap: Map[String, Int] = getCountedMap(sequences)

    new NGramAnnotatorModel()
      .setHistories(historiesMap)
      .setSequences(sequencesMap)
  }

  def getCountedMap(sequence: Seq[Annotation]) = {
    sequence
      .groupBy[String](annotation => annotation.result)
      .map { case (key, values) => (key, values.size) }
  }

  def getTransformedNGramString(results: Seq[String], n: Int): Seq[Annotation] = {
    val pipelineModel = getNGramModel(n)
    val annotated: Map[String, Seq[Annotation]] = new LightPipeline(pipelineModel).fullAnnotate(results.head)

    annotated(n + "ngrams")
  }

  def getNGramModel(n: Int): PipelineModel = {
    ResourceHelper.spark

    import ResourceHelper.spark.implicits._

    val nlpPipeline = new Pipeline()

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val nGramGenerator = new NGramGenerator()
      .setInputCols("token")
      .setOutputCol(s"$n" + "ngrams")
      .setN(n)
      .setEnableCumulative(false)

    nlpPipeline.setStages(Array(documentAssembler, tokenizer, nGramGenerator))

    nlpPipeline.fit(Seq.empty[String].toDF("text"))

  }

  override val description: String = "NGrammAnnotator"
}
