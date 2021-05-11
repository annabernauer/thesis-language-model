package com.mailytica.thesis.language.model.evaluationLM.annotators

import com.johnsnowlabs.nlp.AnnotatorType.TOKEN
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel}
import com.mailytica.thesis.language.model.evaluationLM.annotators.ShortDocumentsFilter.DefaultMinTokensParam
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}

class ShortDocumentsFilter(override val uid: String) extends AnnotatorModel[ShortDocumentsFilter] with DefaultParamsWritable {

  override val outputAnnotatorType: AnnotatorType = TOKEN

  override val inputAnnotatorTypes: Array[String] = Array(TOKEN)

  def this() = this(Identifiable.randomUID("SMALL_DOCUMENTS_CLEANER"))

  setDefault(inputCols, Array(TOKEN))
  setDefault(outputCol, s"${TOKEN}_${uid}")

  val minTokensParam: Param[Int] = new Param[Int](this, "minTokensParam", "minimum number of tokens")

  def setMinTokensParam(value: Int): ShortDocumentsFilter = set(minTokensParam, value)

  def getMinTokensParam: Int = $(minTokensParam)

  setDefault(minTokensParam, DefaultMinTokensParam)

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {

    annotations.size match {
      case smallSize if smallSize < this.getMinTokensParam => Seq.empty
      case _ => annotations
    }
  }
}

object ShortDocumentsFilter extends DefaultParamsReadable[RedundantTextTrimmer] {
  val DefaultMinTokensParam = 5

}
