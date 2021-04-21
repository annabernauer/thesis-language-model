package com.mailytica.thesis.language.model.ngram.pipelines.matching

import com.mailytica.thesis.language.model.ngram.annotators.WordLengthMatcherModel
import org.apache.spark.ml.PipelineStage

object WordLengthMatching extends AbstractMatching {

  override def getSpecificStages(): Array[_ <: PipelineStage] = {
    val wordLengthMatcherModel = new WordLengthMatcherModel()
      .setInputCols("token")
      .setOutputCol("filteredWordsByLength")

    Array(wordLengthMatcherModel)
  }
}
