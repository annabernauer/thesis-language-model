package com.mailytica.thesis.language.model.evaluation.annotators

import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel}
import com.johnsnowlabs.nlp.AnnotatorType.DOCUMENT
import org.apache.spark.ml.util.Identifiable

import scala.util.matching.Regex

class RedundantTextTrimmer(override val uid: String) extends AnnotatorModel[RedundantTextTrimmer]{

  override val outputAnnotatorType: AnnotatorType = DOCUMENT

  override val inputAnnotatorTypes: Array[String] = Array(DOCUMENT)

  def this() = this(Identifiable.randomUID("SENTENCE_END_MARKER"))

  val REGEX_FOOTER : String = "^(?i)(Mit freundlichen Grüßen|Viele Grüße|Beste Grüße|Liebe Grüße|freundliche Grüße)[^*]*"

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {

    annotations.map{

      annotation =>
        val result = annotation.result
        val trimmedResult = result.replaceAll(REGEX_FOOTER, "")

        annotation.copy(result = trimmedResult)
        }
  }

}
