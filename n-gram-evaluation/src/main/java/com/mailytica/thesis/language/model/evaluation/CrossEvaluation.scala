package com.mailytica.thesis.language.model.evaluation

import com.johnsnowlabs.nlp.Annotation
import com.johnsnowlabs.nlp.util.io.ResourceHelper.spark.sqlContext
import com.mailytica.thesis.language.model.evaluation.pipelines.NGramSentencePrediction.getStages
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.{DataFrame, SparkSession}

object CrossEvaluation {

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
    val n = 8
    nlpPipeline.setStages(getStages(n))

    val path = "src/main/resources/sentencePrediction/textsForTraining/bigData/nGramTesting/messagesSmall.csv"

    val df: DataFrame = sqlContext.read.format("com.databricks.spark.csv")
      .option("header", "true")
      .option("quote", "\"")
      .option("escape", "\\")
      .option("multiLine", value = true)
      .load(path)

    val splitArray = df.randomSplit(Array(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1))

    val testData : DataFrame = splitArray(0)
    val trainingData: DataFrame = splitArray
      .diff(Array(testData))                                                        //remove
      .reduce(_ union _)

    testData.show()
    println("#############################")
    trainingData.show()

    val pipelineModel: PipelineModel = nlpPipeline.fit(trainingData.toDF("text"))
    val annotated: DataFrame = pipelineModel.transform(testData.toDF("text"))

    val processed = annotated
      .select("sentencePrediction")
      .cache()

    processed.show(100, false)

    val annotationsPerDocuments: Array[Annotation] = processed
      .as[Array[Annotation]]
      .collect()
      .flatten

    printCalculations(annotationsPerDocuments, n, 0)

//    List.range(0, splitArray.length).foreach { index =>
//      val testData : DataFrame = splitArray(index)
//      val trainingData: DataFrame = splitArray
//        .diff(Array(testData))                                                        //remove
//        .reduce(_ union _)
//
//      val pipelineModel: PipelineModel = nlpPipeline.fit(trainingData.toDF("text"))
//      val annotated: DataFrame = pipelineModel.transform(testData.toDF("text"))
//
//      val processed = annotated
//        .select("sentencePrediction")
//        .cache()
//
//      processed.show(100, false)
//
//      val annotationsPerDocuments: Array[Annotation] = processed
//        .as[Array[Annotation]]
//        .collect()
//        .flatten
//
//      printCalculations(annotationsPerDocuments, n, index)
//
//    }



  }
  def printCalculations(annotationsPerDocuments: Array[Annotation], n: Int, index: Int) = {
    val avgLogLikelihoodAverage = getAverage("avgLogLikelihood", annotationsPerDocuments)
    val durationAverage = getAverage("duration", annotationsPerDocuments)
    val perplexityAverage = getAverage("perplexity", annotationsPerDocuments)
    val medianAverage = getAverage("medianLikelihoods", annotationsPerDocuments)
    val avgLikelihood = getAverage("avgLikelihood", annotationsPerDocuments)

    println(s"index $index")
    println(s"n = $n")
    println("avgLogLikelihoodAverage \t" + avgLogLikelihoodAverage)
    println("duration avg \t\t\t\t" + durationAverage)
    println("perplexityAverage \t\t\t" + perplexityAverage)
    println("median avg \t\t\t\t\t" + medianAverage)
    println("avgLikelihood \t\t\t\t" + avgLikelihood)
    println(s"documentscount ${annotationsPerDocuments.length}")
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
