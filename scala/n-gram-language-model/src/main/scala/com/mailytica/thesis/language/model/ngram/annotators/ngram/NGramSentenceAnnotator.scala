package com.mailytica.thesis.language.model.ngram.annotators.ngram

import com.codahale.metrics.Timer
import com.johnsnowlabs.nlp.AnnotatorApproach
import com.johnsnowlabs.nlp.AnnotatorType.TOKEN
import com.mailytica.thesis.language.model.ngram.Timer.{ngramSentenceTimerTrain, ngramTimerTrain}
import org.apache.commons.lang.time.StopWatch
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.Dataset

import java.util.concurrent.TimeUnit

class NGramSentenceAnnotator (override val uid: String) extends AnnotatorApproach[NGramSentenceAnnotatorModel]{
  override val description: String = "NGRAM_SENTENCE_ANNOTATOR"

  override val inputAnnotatorTypes: Array[String] = Array(TOKEN)
  override val outputAnnotatorType: AnnotatorType = TOKEN

  val n: Param[Int] = new Param(this, "n", "")

  def setN(value: Int): this.type = set(this.n, value)

  setDefault(this.n, 3)

  def this() = this(Identifiable.randomUID("NGRAM_SENTENCES"))

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): NGramSentenceAnnotatorModel = {
    val stopwatch = new StopWatch
    stopwatch.reset()
    stopwatch.start()

    val nGramAnnotator = new NGramAnnotator()
      .setInputCols("token")
      .setOutputCol("ngramsProbability")
      .setN($(n))

    stopwatch.split()

    val model = nGramAnnotator.train(dataset)

//    ngramTimerTrain.update(stopwatch.getTime - stopwatch.getSplitTime, TimeUnit.MILLISECONDS)
    ngramSentenceTimerTrain.update(stopwatch.getTime, TimeUnit.MILLISECONDS)

    new NGramSentenceAnnotatorModel()
      .setNGramAnnotatorModel(model)
      .setN($(n))
  }
}
