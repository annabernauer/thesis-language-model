package com.mailytica.thesis.language.model.evaluation

import com.johnsnowlabs.nlp.Annotation
import com.johnsnowlabs.nlp.util.io.ResourceHelper.spark.sqlContext
import com.mailytica.thesis.language.model.evaluation.pipelines.NGramSentencePrediction.getStages
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.{DataFrame, SparkSession}

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

    val path = "src/main/resources/sentencePrediction/textsForTraining/bigData/nGramTesting/messages.csv"

    val df: DataFrame = sqlContext.read.format("com.databricks.spark.csv")
      .option("header", "true")
      .option("quote", "\"")
      .option("escape", "\\")
      .option("multiLine", value = true)
      .load(path)

    val splitArray = df.randomSplit(Array(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1))

    val allCrossFoldValues: List[(Double, Double, Double, Double, Double)] = List.range(0, 1).map { index =>
      val testData: DataFrame = splitArray(index)
      val trainingData: DataFrame = splitArray
        .diff(Array(testData)) //remove
        .reduce(_ union _)

      //      testData.show()
      //      println("#############################")
      //      trainingData.show()

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

      getCalculations(annotationsPerDocuments, index)
    }
    val crossFoldAverage : (Double, Double, Double, Double, Double) =
      allCrossFoldValues.reduce((a,b) =>
        ((a._1 + b._1) / splitArray.length.toDouble,
        (a._2 + b._2) / splitArray.length.toDouble,
        (a._3 + b._3) / splitArray.length.toDouble,
        (a._4 + b._4) / splitArray.length.toDouble,
        (a._5 + b._5) / splitArray.length.toDouble))

    val mapValues : Map[String, Double] = Map("avgLogLikelihoodAverage" -> crossFoldAverage._1,
      "durationAverage" -> crossFoldAverage._2,
      "perplexityAverage" -> crossFoldAverage._3,
      "medianAverage" -> crossFoldAverage._4,
      "avgLikelihood" -> crossFoldAverage._5)

    println("#######################################################################")
    println(s"n = $n")
    mapValues.foreach(value => println(s"${value._1} ${value._2}"))

  }

  def getCalculations(annotationsPerDocuments: Array[Annotation], index: Int) = {
    val avgLogLikelihoodAverage = getAverage("avgLogLikelihood", annotationsPerDocuments)
    val durationAverage = getAverage("duration", annotationsPerDocuments)
    val perplexityAverage = getAverage("perplexity", annotationsPerDocuments)
    val medianAverage = getAverage("medianLikelihoods", annotationsPerDocuments)
    val avgLikelihood = getAverage("avgLikelihood", annotationsPerDocuments)

    println("###########################################################")
    println(s"index $index")
    println("avgLogLikelihoodAverage \t" + avgLogLikelihoodAverage)
    println("duration avg \t\t\t\t" + durationAverage)
    println("perplexityAverage \t\t\t" + perplexityAverage)
    println("median avg \t\t\t\t\t" + medianAverage)
    println("avgLikelihood \t\t\t\t" + avgLikelihood)
    println(s"documentsCount ${annotationsPerDocuments.length}")

    val calculations = (avgLogLikelihoodAverage,
      durationAverage,
      perplexityAverage,
      medianAverage,
      avgLikelihood)

    calculations
  }


  def getAverage(key: String, annotationsPerDocuments: Array[Annotation]) = {
    annotationsPerDocuments
      .map(language_model_annotation =>
        language_model_annotation
          .metadata
          .getOrElse(key, "0.0").toDouble
      ).filterNot(value => value.isNaN)
      .sum / annotationsPerDocuments.length
  }

}
