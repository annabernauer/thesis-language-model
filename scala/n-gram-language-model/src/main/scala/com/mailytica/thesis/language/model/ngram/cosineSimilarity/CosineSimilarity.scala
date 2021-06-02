package com.mailytica.thesis.language.model.ngram.cosineSimilarity

import com.mailytica.thesis.language.model.ngram.Timer.{cosineDotProduct, cosineNormASqurt, cosineNormBSqurt, cosineSimilarityTimer, stopwatch}
import com.mailytica.thesis.language.model.ngram.cosineSimilarity.pipelines.CosineSimilarityPipelines.getVectorizerStages
import org.apache.commons.lang.time.StopWatch
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.{col, udf}

import java.util.concurrent.TimeUnit
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.{Await, Future}
import scala.concurrent.duration.Duration

object CosineSimilarity {

  def vectorizeData(df: DataFrame, predictionInputCol: String, referenceInputCol: String, needsDocAssembl: Boolean) = {

    val vectorizePipeline = new Pipeline()
    vectorizePipeline.setStages(
      getVectorizerStages(predictionInputCol, "prediction", needsDocAssembl) ++
        getVectorizerStages(referenceInputCol, "reference", needsDocAssembl))

    val pipelineModel: PipelineModel = vectorizePipeline.fit(df)
    val annotatedHypothesis: DataFrame = pipelineModel.transform(df)

    val withCosineColumn: DataFrame = annotatedHypothesis.withColumn("cosine", cosineSimilarityUdf(col("vectorizedCount_prediction"), col("vectorizedCount_reference")))
    withCosineColumn
  }

  val cosineSimilarityUdf : UserDefinedFunction = udf{ (vectorA : Vector, vectorB: Vector) =>
    cosineSimilarity(vectorA, vectorB) //cosineSimilarity of each row
  }

  def cosineSimilarity(vectorA: Vector, vectorB: Vector) : Double = {
    val stopwatch2 = new StopWatch
    stopwatch2.start()

    stopwatch.reset()
    stopwatch.start()

    val vectorArrayA = vectorA.toArray
    val vectorArrayB = vectorB.toArray

    val normASqrt : Future[Double] = Future {
      Math.sqrt(vectorArrayA.map { value =>
        Math.pow(value, 2)
      }.sum)
    }

    cosineNormASqurt.update(stopwatch.getTime, TimeUnit.MILLISECONDS)
    stopwatch.reset()
    stopwatch.start()

    val normBSqrt : Future[Double] = Future {
      Math.sqrt(vectorArrayB.map { value =>
        Math.pow(value, 2)
      }.sum)
    }

    cosineNormBSqurt.update(stopwatch.getTime, TimeUnit.MILLISECONDS)
    stopwatch.reset()
    stopwatch.start()

    val dotProduct : Future[Double] =  Future {
      vectorArrayA
        .zip(vectorArrayB)
        .map { case (x, y) => x * y }
        .sum
    }

    cosineDotProduct.update(stopwatch.getTime, TimeUnit.MILLISECONDS)

    val cosinusFuture = normASqrt
      .zip(normBSqrt)
      .zip(dotProduct)
      .map{
        case ((normASqrt, normBSqrt), dotProduct) =>  {
          val div : Double = normASqrt * normBSqrt
          if( div == 0 )
            0
          else
            dotProduct / div
        }
      }

    val result = Await.result(cosinusFuture, Duration(1, TimeUnit.MINUTES))

    cosineSimilarityTimer.update(stopwatch2.getTime, TimeUnit.MILLISECONDS)
    result
  }


  def calculateCosineValues(vectorizedData: DataFrame, predictionInputCol: String, referenceInputCol: String, spark: SparkSession) = {

    import spark.implicits._

    vectorizedData.select("seeds", predictionInputCol, referenceInputCol, "ngrams_reference", "ngrams_prediction", "cosine").show(20,false)
    //        writeToFile(vectorizedData, fold)

    val cosineValues = vectorizedData
      .select("cosine")
      .as[Double]
      .collect()

    val crossfoldAverage = (cosineValues.sum) / cosineValues.length

    (cosineValues, crossfoldAverage)
  }

}
