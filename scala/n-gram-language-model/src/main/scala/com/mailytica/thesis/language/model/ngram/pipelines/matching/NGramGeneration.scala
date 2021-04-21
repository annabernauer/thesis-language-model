package com.mailytica.thesis.language.model.ngram.pipelines.matching

import com.johnsnowlabs.nlp.annotators.NGramGenerator
import com.mailytica.thesis.language.model.ngram.annotators.ngram.NGramCustomGenerator
import org.apache.spark.ml.PipelineStage

object NGramGeneration extends AbstractMatching {

  override def getSpecificStages(): Array[_ <: PipelineStage] = {
    val nGramGenerator = new NGramCustomGenerator()
      .setInputCols("token")
      .setOutputCol("ngrams")
      .setN(2)
      .setDelimiter("_")

    Array(nGramGenerator)
  }
}
