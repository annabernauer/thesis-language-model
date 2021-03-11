package com.mailytica.thesis.language.model.ngram.regexDemo

import com.johnsnowlabs.nlp.DocumentAssembler
import com.johnsnowlabs.nlp.annotator.Tokenizer
import org.apache.spark.ml.PipelineStage

abstract class AbstractMatching {

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

}
