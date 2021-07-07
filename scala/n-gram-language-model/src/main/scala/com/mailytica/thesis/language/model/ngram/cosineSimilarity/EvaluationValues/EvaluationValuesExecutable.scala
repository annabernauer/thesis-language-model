package com.mailytica.thesis.language.model.ngram.cosineSimilarity.EvaluationValues

import breeze.numerics.sqrt
import com.mailytica.thesis.language.model.ngram.cosineSimilarity.DeepLearningCosineExecutable.{emb, epo, n, srcName}

import java.io.{BufferedReader, File, FileInputStream, InputStreamReader}
import scala.collection.immutable
import com.mailytica.thesis.language.model.util.Utility.{printToFile, srcName}

object EvaluationValuesExecutable {

  def main(args: Array[String]): Unit = {
    println("starting")

    val ngramValues: Seq[(Int, EvaluationTypes)] = Range.inclusive(3, 9).map { ngram =>
      println(s"ngram $ngram")

      val evaluationFolds: Seq[EvaluationTypes] = Range.inclusive(0, 9).map { fold =>

//                val source = scala.io.Source.fromFile(s"resources/crossFoldValues/messages_n_$ngram/messages_n_${ngram}_fold_$fold/cosineValues")
//                val preLines: Seq[String] = try source.getLines.toList finally source.close()
//                println(preLines.size)

        val reader = new BufferedReader(new InputStreamReader(new FileInputStream(
          s"src/main/resources/crossFoldValues/messages_n_$ngram/messages_n_${ngram}_fold_$fold/cosineValues")))
        val preLines: Seq[String] = Stream.continually(reader.readLine()).takeWhile(_ != null)

        val lines = preLines.drop(1).dropRight(1)

        val crossfoldValues: Seq[Double] = lines.flatMap(s => scala.util.Try(s.toDouble).toOption)

        val minValue = MinValue(crossfoldValues.min)
        val maxValue = MaxValue(crossfoldValues.max)

        val median = Median(crossfoldValues.sortWith(_ < _).drop(crossfoldValues.length / 2).head)

        val avg = Avg(crossfoldValues.sum / crossfoldValues.size)
        val standardDeviation = StandardDeviation(sqrt(crossfoldValues.map(value => Math.pow(value - avg.value, 2)).sum))
        EvaluationTypes(median, minValue, maxValue, standardDeviation, avg)
      }

      val minValueAvgFolds = MinValue(evaluationFolds.map(evaluationFold => evaluationFold.minValue.value).sum / evaluationFolds.size)
      val maxValueAvgFolds = MaxValue(evaluationFolds.map(evaluationFold => evaluationFold.maxValue.value).sum / evaluationFolds.size)
      val medianAvgFolds = Median(evaluationFolds.map(evaluationFold => evaluationFold.median.value).sum / evaluationFolds.size)
      val stdDeviationAvgFolds = StandardDeviation(evaluationFolds.map(evaluationFold => evaluationFold.standardDeviation.value).sum / evaluationFolds.size)
      val avgAvgFolds = Avg(evaluationFolds.map(evaluationFold => evaluationFold.avg.value).sum / evaluationFolds.size)

      (ngram, EvaluationTypes(medianAvgFolds, minValueAvgFolds, maxValueAvgFolds, stdDeviationAvgFolds, avgAvgFolds))
    }

    printToFile(new File(s"target/furtherEvaluationValuesNgram.txt")) { p =>
      ngramValues.foreach { value =>
        p.println("#########################")
        p.println(s"ngram = ${value._1}")
        p.println(s"median = ${value._2.median}")
        p.println(s"minValue = ${value._2.minValue}")
        p.println(s"maxValue = ${value._2.maxValue}")
        p.println(s"stdDeviation = ${value._2.standardDeviation}")
        p.println(s"avg = ${value._2.avg}")
      }
    }
    println("finished")


    println("starting deepLearning")

    val ngramValuesDL: Seq[(Int, EvaluationTypes)] = Range.inclusive(3, 9).map { ngram =>
      println(s"ngram $ngram")

      val evaluationFoldsDL: Seq[EvaluationTypes] = Range.inclusive(0, 9).map { fold =>

        //                val source = scala.io.Source.fromFile(s"resources/crossFoldValues/messages_n_$ngram/messages_n_${ngram}_fold_$fold/cosineValues")
        //                val preLines: Seq[String] = try source.getLines.toList finally source.close()
        //                println(preLines.size)

        val reader = new BufferedReader(new InputStreamReader(new FileInputStream(
          s"src/main/resources/deep_learning_model/messages_n_${ngram}_emb_100_epo_25/messages_n_${ngram}_fold_$fold/cosineValues_deep_learning")))
        val preLines: Seq[String] = Stream.continually(reader.readLine()).takeWhile(_ != null)

        val lines = preLines.drop(1).dropRight(1)

        val crossfoldValues: Seq[Double] = lines.flatMap(s => scala.util.Try(s.toDouble).toOption)

        val minValue = MinValue(crossfoldValues.min)
        val maxValue = MaxValue(crossfoldValues.max)

        val median = Median(crossfoldValues.sortWith(_ < _).drop(crossfoldValues.length / 2).head)

        val avg = Avg(crossfoldValues.sum / crossfoldValues.size)
        val standardDeviation = StandardDeviation(sqrt(crossfoldValues.map(value => Math.pow(value - avg.value, 2)).sum))
        EvaluationTypes(median, minValue, maxValue, standardDeviation, avg)
      }

      val minValueAvgFolds = MinValue(evaluationFoldsDL.map(evaluationFold => evaluationFold.minValue.value).sum / evaluationFoldsDL.size)
      val maxValueAvgFolds = MaxValue(evaluationFoldsDL.map(evaluationFold => evaluationFold.maxValue.value).sum / evaluationFoldsDL.size)
      val medianAvgFolds = Median(evaluationFoldsDL.map(evaluationFold => evaluationFold.median.value).sum / evaluationFoldsDL.size)
      val stdDeviationAvgFolds = StandardDeviation(evaluationFoldsDL.map(evaluationFold => evaluationFold.standardDeviation.value).sum / evaluationFoldsDL.size)
      val avgAvgFolds = Avg(evaluationFoldsDL.map(evaluationFold => evaluationFold.avg.value).sum / evaluationFoldsDL.size)

      (ngram, EvaluationTypes(medianAvgFolds, minValueAvgFolds, maxValueAvgFolds, stdDeviationAvgFolds, avgAvgFolds))
    }

    printToFile(new File(s"target/furtherEvaluationValuesDL.txt")) { p =>
      ngramValuesDL.foreach { value =>
        p.println("#########################")
        p.println(s"ngram = ${value._1}")
        p.println(s"median = ${value._2.median}")
        p.println(s"minValue = ${value._2.minValue}")
        p.println(s"maxValue = ${value._2.maxValue}")
        p.println(s"stdDeviation = ${value._2.standardDeviation}")
        p.println(s"avg = ${value._2.avg}")
      }
    }
    println("finished")
  }
}
