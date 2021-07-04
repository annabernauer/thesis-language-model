package com.mailytica.thesis.language.model.ngram.cosineSimilarity.EvaluationValues

import java.io.File

import scala.collection.immutable
import com.mailytica.thesis.language.model.util.Utility.{printToFile, srcName}

class EvaluationValues {

  def main(args: Array[String]): Unit = {

    val ngramValues: Seq[(Int, EvaluationTypes)] = Range.inclusive(3, 9).map { ngram =>

      val evaluationFolds: Seq[EvaluationTypes] = Range.inclusive(0, 9).map { fold =>

        val source = scala.io.Source.fromFile(s"resources/crossFoldValues/messages_n_$ngram/messages_n_${ngram}_fold_$fold/cosineValues")
        val preLines: Seq[String] = try source.getLines.toList finally source.close()
        println(preLines.size)

        val lines = preLines.drop(1).dropRight(1)
        println(lines.size)

        val crossfoldValues: Seq[Double] = lines.flatMap(s => scala.util.Try(s.toDouble).toOption)

        val minValue = MinValue(crossfoldValues.min)
        val maxValue = MaxValue(crossfoldValues.max)

        val median = Median(crossfoldValues.sortWith(_ < _).drop(crossfoldValues.length / 2).head)
        EvaluationTypes(median, minValue, maxValue)
      }

      val minValueAvgFolds = MinValue(evaluationFolds.map(evaluationFold => evaluationFold.minValue.value).sum / evaluationFolds.size)
      val maxValueAvgFolds = MaxValue(evaluationFolds.map(evaluationFold => evaluationFold.maxValue.value).sum / evaluationFolds.size)
      val medianAvgFolds = Median(evaluationFolds.map(evaluationFold => evaluationFold.median.value).sum / evaluationFolds.size)

      (ngram, EvaluationTypes(medianAvgFolds, minValueAvgFolds, maxValueAvgFolds))
    }

    printToFile(new File(s"target/furtherEvaluationValues.txt")) { p =>
      ngramValues.foreach { value =>
        p.println("#########################")
        p.println(s"ngram = ${value._1}")
        p.println(s"median = ${value._2.median}")
        p.println(s"minValue = ${value._2.minValue}")
        p.println(s"maxValue = ${value._2.maxValue}")
      }
    }
  }
}
