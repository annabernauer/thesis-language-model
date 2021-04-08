package com.mailytica.thesis.language.model.evaluation.annotators.ngram

import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.nlp.util.io.ResourceHelper.spark.sqlContext
import com.johnsnowlabs.nlp.{Annotation, LightPipeline}
import com.mailytica.thesis.language.model.evaluation.pipelines.NGramSentencePrediction.getStages
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{Matchers, WordSpec}
import shapeless.syntax.std.tuple.unitTupleOps

import java.util
import scala.io.{Codec, Source}

@RunWith(classOf[JUnitRunner])
class NGramSentenceEvaluationSpec extends WordSpec with Matchers {

  "A text" when {

    ResourceHelper.spark

    import ResourceHelper.spark.implicits._

    val nlpPipeline = new Pipeline()



    "is trained with more data" when {
      nlpPipeline.setStages(getStages(5))


      val texts: Seq[String] = getCleanResourceText("/sentencePrediction/textsForTraining/productionRelease", 9)

      val texts2: Seq[String] = getCleanResourceText("/sentencePrediction/textsForTraining/shippingNotification", 9)

      //      val pipelineModel: PipelineModel = nlpPipeline.fit((texts ++ texts2).toDF("text"))

      "has a text with matches" should {

        //        val annotated: Seq[Map[String, Seq[Annotation]]] = texts.map(inputString => new LightPipeline(pipelineModel).fullAnnotate(inputString))

        //        val flatMap: Seq[Annotation] = annotated.flatMap(map => map("sentencePrediction"))
        //        flatMap.foreach(annotation => println(s"${annotation.result} ${annotation.metadata}"))

        "have predicted the sentence" in {

        }
      }
    }
    "is trained with big data" when {
      nlpPipeline.setStages(getStages(5))

      val path = "src/main/resources/sentencePrediction/textsForTraining/bigData/messages.csv"

      val df: DataFrame = sqlContext.read.format("com.databricks.spark.csv")
        .option("header", "true")
        .option("quote", "\"")
        .option("escape", "\\")
        .option("multiLine", value = true)
        .load(path)

            df.show()

      //training
            val pipelineModel: PipelineModel = nlpPipeline.fit(df.toDF("text"))
//            pipelineModel.write.overwrite().save("target/pipelineModel")

//      val pipelineModel = PipelineModel.load("target/pipelineModel")

      "has a text with matches" should {

        val annotated: DataFrame = pipelineModel.transform(df.toDF("text"))

        val processed = annotated
          .select("sentencePrediction")
          .cache()

        processed.show(100, false)

        val annotationsPerDocuments: Array[Annotation] = processed
          .as[Array[Annotation]]
          .collect()
          .flatten


        //        annotationsPerDocuments.foreach(println)

        val avgLogLikelihoodAverage = getAverage("avgLogLikelihood", annotationsPerDocuments)
        val durationAverage = getAverage("duration", annotationsPerDocuments)
        val perplexityAverage = getAverage("perplexity", annotationsPerDocuments)
        val medianAverage = getAverage("medianLikelihoods", annotationsPerDocuments)
        val avgLikelihood = getAverage("avgLikelihood", annotationsPerDocuments)

        println("avgLogLikelihoodAverage " + avgLogLikelihoodAverage)
        println("duration avg " + durationAverage)
        println("perplexityAverage " + perplexityAverage)
        println("median avg " + medianAverage)
        println("avgLikelihood " + avgLikelihood)

        "have predicted the sentence" in {

        }
      }
    }
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

  def getCleanResourceText(path: String, quantity: Int) = {
    Seq.range(0, quantity).map {
      x => {
        resource
          .managed(getClass.getResourceAsStream(s"$path/00$x.txt"))
          .acquireAndGet(inputStream => {

            Source
              .fromInputStream(inputStream)(Codec.UTF8)
              .mkString + " <SENTENCE_END>"
          })
      }
    }
  }

}
