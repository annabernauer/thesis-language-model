package com.mailytica.thesis.language.model.evaluation

import com.johnsnowlabs.nlp.{Annotation, DocumentAssembler, Finisher}
import com.johnsnowlabs.nlp.annotator.Tokenizer
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.util.io.ResourceHelper.spark.sqlContext
import org.apache.spark.ml.{Pipeline, PipelineModel, linalg}
import org.apache.spark.ml.feature.CountVectorizer
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.{col, udf}


object CosineSimilarity {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession
      .builder
      .config("spark.driver.maxResultSize", "5g")
      .config("spark.driver.memory", "12g")
      .config("spark.sql.codegen.wholeStage", "false") // deactivated as the compiled grows to big (exception)
      .master(s"local[3]")
      .getOrCreate()

    import spark.implicits._

    //    val reference = "Sollten Sie weitere Fragen haben können Sie mich gerne kontaktieren."
    val reference = Seq("Ihnen den neuen Korrekturabzug. Anbei sende", "Anbei sende ich Ihnen den").toDF("text")
    val hypothese = Seq("Anbei sende ich Ihnden den").toDF("text")
    val context = Seq("Anbei sende ich").toDF("text")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")
      .setCleanupMode("disabled")

    val sentenceDetector = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentences")

    val tokenizer = new Tokenizer()
      .setInputCols("sentences")
      .setOutputCol("tokens")

    val finisher = new Finisher()
      .setInputCols("tokens")
      .setOutputCols("finishedTokens")

    val countVector = new CountVectorizer()
      .setInputCol("finishedTokens")
      .setOutputCol("vectorizedCount")

    val nlpPipeline = new Pipeline()
    nlpPipeline.setStages(Array(documentAssembler, sentenceDetector, tokenizer, finisher, countVector))

    val pipelineModel: PipelineModel = nlpPipeline.fit(reference.union(hypothese))
    val annotatedReference: DataFrame = pipelineModel.transform(reference)
    val annotatedHypothese: DataFrame = pipelineModel.transform(hypothese)

    annotatedHypothese.show()
//    annotatedReference.join(annotatedHypothese, )

    val withCosineColumn: DataFrame = annotatedHypothese.withColumn("cosine", cosineSimilarityUdf(col("vectorizedCount"), col("vectorizedCount")))

    withCosineColumn.show(false)

  }

  val cosineSimilarityUdf : UserDefinedFunction = udf{ (vectorA : Vector, vectorB: Vector) =>
    cosineSimilarity(vectorA, vectorB)
  }

  def cosineSimilarity(vectorA: Vector, vectorB: Vector) : Double = {

    val vectorArrayA = vectorA.toArray
    val vectorArrayB = vectorB.toArray

    val normASqrt : Double = Math.sqrt(vectorArrayA.map{ value =>
      Math.pow(value , 2)
    }.sum)

    val normBSqrt : Double = Math.sqrt(vectorArrayB.map{ value =>
      Math.pow(value , 2)
    }.sum)

    val dotProduct : Double =  vectorArrayA
      .zip(vectorArrayB)
      .map{case (x,y) => x*y }
      .sum

    val div : Double = normASqrt * normBSqrt
    if( div == 0 )
      0
    else
      dotProduct / div
  }

}
