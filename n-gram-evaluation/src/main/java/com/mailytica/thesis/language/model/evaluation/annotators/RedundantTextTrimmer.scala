package com.mailytica.thesis.language.model.evaluation.annotators

import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel}
import com.johnsnowlabs.nlp.AnnotatorType.DOCUMENT
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}

import scala.util.matching.Regex

class RedundantTextTrimmer(override val uid: String) extends AnnotatorModel[RedundantTextTrimmer] with DefaultParamsWritable {

  override val outputAnnotatorType: AnnotatorType = DOCUMENT

  override val inputAnnotatorTypes: Array[String] = Array(DOCUMENT)

  def this() = this(Identifiable.randomUID("REDUNDANT_TEXT_TRIMMER"))

  val REGEX_FOOTER : String = "(?i)(Mit freundlichen Grüßen|Viele Grüße|Beste Grüße|Liebe Grüße|freundliche Grüße|With kind regards)[^*]*" //Doesn't work with ^ for occurrences on the start of a line

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {

    annotations.map{

      annotation =>
        val result : String = annotation.result
        val trimmedResult : String = result.replaceAll(REGEX_FOOTER, "")

        annotation.copy(result = trimmedResult)

        }
  }

}

object RedundantTextTrimmer extends DefaultParamsReadable[RedundantTextTrimmer] {

}
