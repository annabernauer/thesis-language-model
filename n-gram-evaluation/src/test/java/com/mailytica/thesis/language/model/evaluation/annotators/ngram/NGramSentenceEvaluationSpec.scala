package com.mailytica.thesis.language.model.evaluation.annotators.ngram

import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.nlp.util.io.ResourceHelper.spark.sqlContext
import com.johnsnowlabs.nlp.{Annotation, LightPipeline}
import com.mailytica.thesis.language.model.evaluation.pipelines.NGramSentencePrediction.getStages
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.DataFrame
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{Matchers, WordSpec}

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
      nlpPipeline.setStages(getStages(4))

      val path = "src/main/resources/sentencePrediction/textsForTraining/bigData/messagesSmall.csv"

      val df: DataFrame = sqlContext.read.format("com.databricks.spark.csv")
        .option("header", "true")
        .option("quote", "\"")
        .option("escape", "\\")
        .option("multiLine", value = true)
        .load(path)

      df.show()

      val pipelineModel: PipelineModel = nlpPipeline.fit(df.toDF("text"))

      "has a text with matches" should {

//        val annotated: Seq[Map[String, Seq[Annotation]]] = strings.map(inputString => new LightPipeline(pipelineModel).fullAnnotate(inputString)) // model transform
//        val flatMap: Seq[Annotation] = annotated.flatMap(map => map("sentencePrediction"))
//        flatMap.foreach(annotation => println(s"${annotation.result} ${annotation.metadata}"))

        val annotated: DataFrame =  pipelineModel.transform(df.toDF("text"))

        annotated.select("sentencePrediction").show(100, false)

        "have predicted the sentence" in {

        }
      }
    }
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
