package com.mailytica.thesis.language.model.ngram.pipelines.matching

import com.johnsnowlabs.nlp.annotators.NGramGenerator
import org.apache.spark.ml.PipelineStage

object NGramGeneration extends AbstractMatching {

  override def getSpecificStages(): Array[_ <: PipelineStage] = {
    val nGramGenerator = new NGramGenerator()
      .setInputCols("token")
      .setOutputCol("ngrams")
      .setN(2)
      .setEnableCumulative(true)
      .setDelimiter("_")

    Array(nGramGenerator)
  }
}
