package com.mailytica.thesis.language.model.ngram.regexDemo

import com.mailytica.thesis.language.model.ngram.Annotator.WordLengthMatcherModel
import org.apache.spark.ml.PipelineStage

object WordLengthMatching extends AbstractMatching {

  override def getSpecificStages(): Array[_ <: PipelineStage] = {
    val wordLengthMatcherModel = new WordLengthMatcherModel()
      .setInputCols("token")
      .setOutputCol("matchedText")

    Array(wordLengthMatcherModel)
  }
}
