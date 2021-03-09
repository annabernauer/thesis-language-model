package com.mailytica.thesis.language.model.ngram.regexDemo

import java.io.{BufferedWriter, File, FileWriter}

import com.johnsnowlabs.nlp.DocumentAssembler
import com.johnsnowlabs.nlp.annotator.Tokenizer
import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage}
import org.apache.spark.sql.{DataFrame, SparkSession}

abstract class AbstractMatching {

  val textList = Seq(
    """Instacart has raised the most valuable private companies in the U.S., n a new funding round led by DST
      |Global and General Catalyst. The round increases Instacart’s valuation to $13.7 billion, up from $8
      |billion when it last raised money in 2018. John Doe, followed by leader Quantum test million hundred""".stripMargin)

  val sparkSession = SparkSession
    .builder
    .config("spark.sql.codegen.wholeStage", "false") // deactivated as the compiled grows to big (exception)
    .master(s"local[1]")
    .getOrCreate()

  import sparkSession.implicits._

  def writeToFile(pathname: String, arrayToWrite: Array[String]): File = {
    val file: File = new File(pathname)
    val bw = new BufferedWriter(new FileWriter(file))

    for (regexRule <- arrayToWrite) {
      bw.write(regexRule)
      bw.write("\n")
    }

    bw.close()
    file
  }

  def getGeneralStages() : Array[_ <: PipelineStage] = {
    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    Array(documentAssembler, tokenizer)
  }

  def getSpecificStages() : Array[_ <: PipelineStage]

  def createDataframe(): DataFrame = {
    val nlpPipeline = new Pipeline()

    nlpPipeline.setStages(getGeneralStages() ++ getSpecificStages())

    val data = textList.toDF("text")

    val pipelineModel: PipelineModel = nlpPipeline.fit(data)

    val nlpData: DataFrame = pipelineModel.transform(data)

    nlpData
  }
}
