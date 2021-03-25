package com.mailytica.thesis.language.model.ngram.pipelines.matching

import com.mailytica.thesis.language.model.ngram.annotators.ngram.NGramAnnotator
import org.apache.spark.ml.PipelineStage

object NGramProbability extends AbstractMatching {

  override def getSpecificStages(): Array[_ <: PipelineStage] = {
    val nGramAnnotator = new NGramAnnotator()
      .setInputCols("token")
      .setOutputCol("ngramsProbability")

    Array(nGramAnnotator)
  }
}
