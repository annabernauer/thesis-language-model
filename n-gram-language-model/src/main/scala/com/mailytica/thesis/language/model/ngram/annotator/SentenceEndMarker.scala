package com.mailytica.thesis.language.model.ngram.annotator

import com.johnsnowlabs.nlp.AnnotatorType.DOCUMENT
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel}
import com.johnsnowlabs.nlp.annotators.NGramGenerator
import org.apache.spark.ml.util.Identifiable

class SentenceEndMarker(override val uid: String) extends AnnotatorModel[NGramGenerator] {

  override val outputAnnotatorType: AnnotatorType = DOCUMENT

  override val inputAnnotatorTypes: Array[String] = Array(DOCUMENT)

  def this() = this(Identifiable.randomUID("SENTENCE_END_MARKER"))

  val SENTENCE_END: String = " <SENTENCE_END>"

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {

    val b = new StringBuilder()

//    SENTENCE_END ++ annotations.headOption

    annotations.map{ sentence =>

      val result = sentence.result.replace(".", "." + SENTENCE_END)

      sentence.copy(

        result = result
      )
    }
  }

}
