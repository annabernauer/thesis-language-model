package com.mailytica.thesis.language.model.evaluation.annotators.ngram

import com.johnsnowlabs.nlp.{Annotation, AnnotatorApproach}
import com.johnsnowlabs.nlp.AnnotatorType.TOKEN
import com.johnsnowlabs.nlp.annotator.NGramGenerator
import com.mailytica.thesis.language.model.util.Utility.{DELIMITER, printToFile}
import org.apache.commons.io.FileUtils
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.Dataset

import java.io.File

class NGramEvaluation (override val uid: String) extends AnnotatorApproach[NGramEvaluationModel]{

  override val description: String = "NGramEvaluation"

  override val inputAnnotatorTypes: Array[String] = Array(TOKEN)
  override val outputAnnotatorType: AnnotatorType = TOKEN

  val n: Param[Int] = new Param(this, "n", "")

  def setN(value: Int): this.type = set(this.n, value)

  setDefault(this.n, 3)

  def this() = this(Identifiable.randomUID("NGRAM_ANNOTATOR"))


  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): NGramEvaluationModel = {
    import dataset.sparkSession.implicits._

    val tokensPerDocuments: Seq[Array[Annotation]] = dataset
      .select(getInputCols.head)
      .as[Array[Annotation]]
      .collect()
      .toSeq

    val dictionary: Set[String] = tokensPerDocuments
      .flatten
      .map(token => token.result)
      .toSet

    val histories: Seq[Annotation] = getTransformedNGramString(tokensPerDocuments, $(n) - 1)
    val sequences: Seq[Annotation] = getTransformedNGramString(tokensPerDocuments, $(n))

//    sequences.foreach(println)

    val historiesMap: Map[String, Int] = getCountedMap(histories)
    val sequencesMap: Map[String, Int] = getCountedMap(sequences)

    val file : File = new File("target\\sentencePrediction\\")
    FileUtils.deleteQuietly(file)
    if (!file.exists) {
      file.mkdir
    }

    printToFile(new File("target\\sentencePrediction\\dictionary.txt")) { p =>
      dictionary.foreach(p.println)
    }
    printToFile(new File("target\\sentencePrediction\\historiesMap.txt")) { p =>
      historiesMap.foreach(p.println)
    }
    printToFile(new File("target\\sentencePrediction\\sequencesMap.txt")) { p =>
      sequencesMap.foreach(p.println)
    }
    printToFile(new File("target\\sentencePrediction\\sequencesKeys.txt")) { p =>
      sequencesMap.keys.foreach(p.println)
    }
    print("INFO: Files were created\n")

    new NGramEvaluationModel()
      .setHistories(historiesMap)
      .setSequences(sequencesMap)
      .setN($(n))
      .setDictionary(dictionary)
  }

  def getCountedMap(sequence: Seq[Annotation]) = {
    sequence
      .groupBy[String](annotation => annotation.result)
      .map { case (key, values) => (key, values.size) }
  }

  def getTransformedNGramString(tokensPerDocuments: Seq[Array[Annotation]], n: Int): Seq[Annotation] = {

    val nGramModel = new NGramCustomGenerator()
      .setInputCols("tokens")
      .setOutputCol(s"$n" + "ngrams")
      .setN(n)
//      .setEnableCumulative(false)
      .setDelimiter(DELIMITER)

    tokensPerDocuments.flatMap { tokens =>
      nGramModel.annotate(tokens)
    }
  }

}
