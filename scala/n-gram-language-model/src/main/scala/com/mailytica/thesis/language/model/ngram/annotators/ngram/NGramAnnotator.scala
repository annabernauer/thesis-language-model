package com.mailytica.thesis.language.model.ngram.annotators.ngram

import java.io.File
import com.johnsnowlabs.nlp.AnnotatorType.TOKEN
import com.johnsnowlabs.nlp.annotator.NGramGenerator
import com.johnsnowlabs.nlp.{Annotation, AnnotatorApproach}
import com.mailytica.thesis.language.model.ngram.annotators.ngram.NGramAnnotator.fold
import com.mailytica.thesis.language.model.ngram.cosineSimilarity.CosineExecutable.n
import org.apache.commons.io.FileUtils
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.Dataset
import com.mailytica.thesis.language.model.util.Utility.{DELIMITER, printToFile, srcName}

class NGramAnnotator(override val uid: String) extends AnnotatorApproach[NGramAnnotatorModel] {

  override val inputAnnotatorTypes: Array[String] = Array(TOKEN)
  override val outputAnnotatorType: AnnotatorType = TOKEN

  val n: Param[Int] = new Param(this, "n", "")

  def setN(value: Int): this.type = set(this.n, value)

  setDefault(this.n, 3)

  def this() = this(Identifiable.randomUID("NGRAM_ANNOTATOR"))

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): NGramAnnotatorModel = {

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

    val historiesMap: Map[String, Int] = getCountedMap(histories)
    val sequencesMap: Map[String, Int] = getCountedMap(sequences)

    val dirCrossfoldName = s"${srcName}_n_${$(n)}"
    val directory = new File(s"target/crossFoldValues/$dirCrossfoldName/${dirCrossfoldName}_fold_${fold}/historiesAndSequences")
    if (!directory.exists) {
      directory.mkdirs
    }

    printToFile(new File(s"$directory/dictionary.txt")) { p =>
      dictionary.foreach(p.println)
    }
    printToFile(new File(s"$directory/historiesMap.txt")) { p =>
      historiesMap.foreach(p.println)
    }
    printToFile(new File(s"$directory/sequencesMap.txt")) { p =>
      sequencesMap.foreach(p.println)
    }
    printToFile(new File(s"$directory/sequencesKeys.txt")) { p =>
      sequencesMap.keys.foreach(p.println)
    }

    println("INFO: Files were created")

    fold = fold + 1

    new NGramAnnotatorModel()
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
      .setNGramMinimum(n)                                                                                               //to get only ngrams, not n-i grams

    tokensPerDocuments.flatMap { tokens =>
      nGramModel.annotate(tokens)
    }
  }

  def printToFile(f: java.io.File)(op: java.io.PrintWriter => Unit) {
    val p = new java.io.PrintWriter(f)
    try { op(p) } finally { p.close() }
  }

  override val description: String = "NGramAnnotator"
}

object NGramAnnotator {
  var fold = 0
}