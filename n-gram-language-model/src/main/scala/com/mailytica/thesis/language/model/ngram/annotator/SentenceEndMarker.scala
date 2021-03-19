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

    //        annotations.map(annotation =>
    //          Annotation(annotation.annotatorType, annotation.begin, annotation.`end` + SENTENCE_END.length -1 , annotation.result.replace(".", SENTENCE_END), annotation.metadata, annotation.embeddings))

    def loop(annotationResult: Seq[Annotation], position: Int = 0, index: Int = 0): Seq[Annotation] = {

      if (annotations.length <= index) {
        return annotationResult
      }

      val annotation = annotations(index)
      val tokenLength = annotation.result.length + SENTENCE_END.length - 1

      loop(
        annotationResult ++ Seq(Annotation(
          annotation.annotatorType,
          position,
          position + tokenLength,
          annotation.result.replace(".", SENTENCE_END),
          annotation.metadata,
          annotation.embeddings)),
        position + tokenLength + 2,
        index + 1)
    }

    loop(Seq.empty)

  }

}
