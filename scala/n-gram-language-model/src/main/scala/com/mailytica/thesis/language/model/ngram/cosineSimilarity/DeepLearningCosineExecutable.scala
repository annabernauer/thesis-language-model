package com.mailytica.thesis.language.model.ngram.cosineSimilarity

import scala.io.Source

object DeepLearningCosineExecutable {

  def main(args: Array[String]): Unit = {

    val bufferedSource = Source.fromFile("src/main/resources/sentencePrediction/deepLearningGeneratedTexts/messages_n_6_emb_100_epo_40/messages_n_6_fold_0/generated_texts")
    val lines = bufferedSource.getLines.toList
    bufferedSource.close

//    lines.map(line => )
    //read file
    //get seed and prediction
//    val predctionMap : List[String, String] = List.empty



    //mapping of prediction and reference
    //
  }

}
