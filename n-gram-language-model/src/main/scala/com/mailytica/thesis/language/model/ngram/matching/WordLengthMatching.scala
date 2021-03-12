package com.mailytica.thesis.language.model.ngram.matching

import com.mailytica.thesis.language.model.ngram.annotator.WordLengthMatcherModel
import org.apache.spark.ml.PipelineStage

object WordLengthMatching extends AbstractMatching {

  override def getSpecificStages(): Array[_ <: PipelineStage] = {
    val wordLengthMatcherModel = new WordLengthMatcherModel()
      .setInputCols("token")
      .setOutputCol("filteredWordsByLength")

    Array(wordLengthMatcherModel)
  }
}
