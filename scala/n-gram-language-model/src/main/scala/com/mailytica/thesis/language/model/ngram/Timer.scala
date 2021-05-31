package com.mailytica.thesis.language.model.ngram

import com.codahale.metrics.{ConsoleReporter, MetricRegistry, Timer}
import org.apache.commons.lang.time.StopWatch

import java.util.concurrent.TimeUnit

object Timer {

  val stopwatch = new StopWatch
  val metricRegistry = new MetricRegistry()
  val ngramTimerTrain: Timer = metricRegistry.timer("NGramAnnotatorModel_train")
  val ngramSentenceTimerTrain: Timer = metricRegistry.timer("NGramSentenceAnnotator_train")
  val ngramSentenceModelTimerAnnotateTimer: Timer = metricRegistry.timer("NGramSentenceAnnotatorModel_annotate")
  val ngramTimerModelAnnotateTimer: Timer = metricRegistry.timer("NGramAnnotatorModel_annotate")
  val cosineSimilarityTimer: Timer = metricRegistry.timer("CosineExecutable_cosineSimilarity")
  val cosineNormASqurt: Timer = metricRegistry.timer("CosineExecutable_normASqrt")
  val cosineNormBSqurt: Timer = metricRegistry.timer("CosineExecutable_normBSqrt")
  val cosineDotProduct: Timer = metricRegistry.timer("CosineExecutable_dotproduct")
  val nGramGeneratorTimer: Timer = metricRegistry.timer("NGramCustomGenerator_annotate")

  val consoleReporter: ConsoleReporter = ConsoleReporter
    .forRegistry(metricRegistry)
    .convertRatesTo(TimeUnit.SECONDS)
    .convertDurationsTo(TimeUnit.MILLISECONDS)
//    .withLoggingLevel(ConsoleReporter.LoggingLevel.DEBUG)
    .build()

}
