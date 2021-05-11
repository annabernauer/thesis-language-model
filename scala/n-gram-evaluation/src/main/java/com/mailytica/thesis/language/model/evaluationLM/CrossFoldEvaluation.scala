package com.mailytica.thesis.language.model.evaluationLM

import com.johnsnowlabs.nlp.Annotation
import com.johnsnowlabs.nlp.util.io.ResourceHelper.spark.sqlContext
import com.mailytica.thesis.language.model.evaluationLM.pipelines.NGramSentencePrediction.getStages
import com.mailytica.thesis.language.model.evaluationLM.returnTypes.{AvgLogLikelihood, Duration, Likelihood, LikelihoodMedian, MetadataTypes, Perplexity, PerplexityMedian}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession, functions}


object CrossFoldEvaluation {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .config("spark.driver.maxResultSize", "5g")
      .config("spark.driver.memory", "12g")
      .config("spark.sql.codegen.wholeStage", "false") // deactivated as the compiled grows to big (exception)
      .master(s"local[3]")
      .getOrCreate()

    import spark.implicits._

    val nlpPipeline = new Pipeline()
    val n = 5
    nlpPipeline.setStages(getStages(n))

    val path = "src/main/resources/sentencePrediction/textsForTraining/messagesSmall.csv"

    val df: DataFrame = sqlContext.read.format("com.databricks.spark.csv")
      .option("header", "true")
      .option("quote", "\"")
      .option("escape", "\\")
      .option("multiLine", value = true)
      .load(path)
//      .orderBy(functions.rand())

    val fraction = 1.0 / 10.toDouble
    val fractionPerSplit = Array.fill(10)(fraction)
    val splitArray: Array[Dataset[Row]] = df.randomSplit(fractionPerSplit)

    val allCrossFoldValues: Array[MetadataTypes] =
      splitArray
        .take(1)
        .map { testData =>

          val trainingData: DataFrame = splitArray
            .diff(Array(testData)) //remove testData
            .reduce(_ union _)

          val pipelineModel: PipelineModel = nlpPipeline.fit(trainingData.toDF("text"))
          val annotated: DataFrame = pipelineModel.transform(testData.toDF("text"))

          val processed = annotated
            .select("sentencePrediction")
            .cache()

          //      processed.show(100, false)

          val annotationsPerDocuments: Array[Annotation] = processed
            .as[Array[Annotation]]
            .collect()
            .flatten
            .filter(annotation => annotation.result != "empty")

//          annotationsPerDocuments.foreach(a => println(a + " " + a.metadata))
          getCalculations(annotationsPerDocuments)
        }


    printAllFolds(allCrossFoldValues)
    println(s"n = $n")

  }

  def getCalculations(annotationsPerDocuments: Array[Annotation]) = {

    val avgLogLikelihoodAverage = AvgLogLikelihood(getAverage("avgLogLikelihood", annotationsPerDocuments))
    val durationAverage = Duration(getAverage("duration", annotationsPerDocuments))
    val likelihoodAverage = Likelihood(getAverage("avgLikelihood", annotationsPerDocuments))
    val likelihoodMedian = LikelihoodMedian(getAverage("medianLikelihoods", annotationsPerDocuments))
    val perplexityAverage = Perplexity(getAverage("perplexity", annotationsPerDocuments))
    val perplexityMedian = PerplexityMedian(medianCalculator(getMapByKey("perplexity", annotationsPerDocuments)))

    val calculations = MetadataTypes(
      avgLogLikelihoodAverage,
      durationAverage,
      likelihoodAverage,
      likelihoodMedian,
      perplexityAverage,
      perplexityMedian
    )

    println("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    calculations.productIterator.foreach(x => println(s"$x"))

    calculations
  }

  def printAllFolds(allCrossFoldValues: Array[MetadataTypes]) = {
    val allFoldsAvgLogLikelihood: AvgLogLikelihood = AvgLogLikelihood(allCrossFoldValues.map(_.avgLogLikelihood.value).sum / allCrossFoldValues.length)
    val allFoldsDuration: Duration = Duration(allCrossFoldValues.map(_.duration.value).sum / allCrossFoldValues.length)
    val allFoldsLikelihood: Likelihood = Likelihood(allCrossFoldValues.map(_.likelihood.value).sum / allCrossFoldValues.length)
    val allFoldsLikelihoodMedian: LikelihoodMedian = LikelihoodMedian(allCrossFoldValues.map(_.likelihoodMedian.value).sum / allCrossFoldValues.length)
    val allFoldsPerplexity: Perplexity = Perplexity(allCrossFoldValues.map(_.perplexity.value).sum / allCrossFoldValues.length)
    val allFoldsPerplexityMedian: PerplexityMedian = PerplexityMedian(allCrossFoldValues.map(_.perplexityMedian.value).sum / allCrossFoldValues.length)

    val allFolds = MetadataTypes(allFoldsAvgLogLikelihood, allFoldsDuration, allFoldsLikelihood ,allFoldsLikelihoodMedian, allFoldsPerplexity, allFoldsPerplexityMedian)

    println("##################################################################################")
    allFolds.productIterator.foreach(println)
  }

  def getMapByKey(key: String, annotationsPerDocuments: Array[Annotation]) = {
    annotationsPerDocuments
      .map(language_model_annotation =>
        language_model_annotation
          .metadata
          .getOrElse(key, "0.0").toDouble
      ).filterNot(value => value.isNaN)
  }


  def getAverage(key: String, annotationsPerDocuments: Array[Annotation]) = {

    getMapByKey(key, annotationsPerDocuments)
      .sum / annotationsPerDocuments.length
  }

  def medianCalculator(seq: Seq[Double]): Double = {
    val sortedSeq: Seq[Double] = seq.sortWith(_ < _)
    if (seq.size % 2 == 1) sortedSeq(sortedSeq.size / 2)
    else {
      val (up: Seq[Double], down: Seq[Double]) = sortedSeq.splitAt(seq.size / 2)
      (up.lastOption.getOrElse(0.0) + down.headOption.getOrElse(0.0)) / 2
    }
  }

}
