package com.mailytica.thesis.language.model.ngram.matching

import com.mailytica.thesis.language.model.ngram.annotator.NGramAnnotator
import org.apache.spark.ml.PipelineStage

object NGramPropability extends AbstractMatching {

  override def getSpecificStages(): Array[_ <: PipelineStage] = {
    val nGramAnnotator = new NGramAnnotator()
      .setInputCols("token")
      .setOutputCol("ngramsProbability")

    Array(nGramAnnotator)
  }
}
