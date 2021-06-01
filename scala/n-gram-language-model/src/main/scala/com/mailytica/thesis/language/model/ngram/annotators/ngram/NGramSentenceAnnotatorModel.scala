package com.mailytica.thesis.language.model.ngram.annotators.ngram

import com.johnsnowlabs.nlp.AnnotatorType.TOKEN
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel}
import com.mailytica.thesis.language.model.ngram.Timer.{ngramSentenceModelTimerAnnotateTimer, stopwatch}
import org.apache.commons.lang.time.StopWatch
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.Identifiable

import java.util.concurrent.TimeUnit
import scala.annotation.tailrec

class NGramSentenceAnnotatorModel(override val uid: String) extends AnnotatorModel[NGramSentenceAnnotatorModel] {

  def this() = this(Identifiable.randomUID("NGRAM_SENTENCES"))

  override val inputAnnotatorTypes: Array[String] = Array(TOKEN)

  override val outputAnnotatorType: AnnotatorType = TOKEN

  val n: Param[Int] = new Param(this, "n", "")

  val SENTENCE_END: String = "<SENTENCE_END>"

  val SENTENCE_START: String = "<SENTENCE_START>"

  val nGramAnnotatorModel: Param[NGramAnnotatorModel] = new Param(this, "nGramAnnotatorModel", "")

  def setN(value: Int): this.type = set(this.n, value)

  def setNGramAnnotatorModel(value: NGramAnnotatorModel): this.type = set(this.nGramAnnotatorModel, value)

  setDefault(this.n, 3)

  // document -> sentence -> token -> ngramannotatorSentenceModel ( ngram -> ngramAnnotatorModel( -> prediction next token) -> predicion sentence)

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {

    stopwatch.reset()
    stopwatch.start()

    @tailrec
    def loop(joinedAnnotations: Seq[Annotation], count: Int = 0): Seq[Annotation] = {

      joinedAnnotations.lastOption match {                                                                      //just for termination condition
        case Some(annotation) =>
          annotation.result match {
            case SENTENCE_END => joinedAnnotations
            case _ => {
              count match {
                case 40 => joinedAnnotations
                case _ => {
                  loop(joinedAnnotations ++ $(nGramAnnotatorModel).annotate(joinedAnnotations), count + 1)
                }
              }
            }
          }
        case None => Seq()
      }
    }

    val prediction = loop(annotations)
//      .filterNot(token => (token.result == SENTENCE_END) || (token.result == SENTENCE_START))          //remove SENTENCE_END and SENTENCE_START tags

    ngramSentenceModelTimerAnnotateTimer.update(stopwatch.getTime, TimeUnit.MILLISECONDS)
    prediction
  }


}
