package com.mailytica.thesis.language.model.ngram.regexDemo

import java.io.{BufferedWriter, File, FileWriter}

import org.apache.spark.sql.{DataFrame, SparkSession}
import com.johnsnowlabs.nlp.{Annotation, DocumentAssembler, LightPipeline}
import com.johnsnowlabs.nlp.annotator.{RegexMatcher, Tokenizer}
import com.johnsnowlabs.nlp.annotators.{RegexMatcherModel, TextMatcher}
import org.apache.spark.ml.{Pipeline, PipelineModel}

object MatchingText extends App {

  val sparkSession = SparkSession
    .builder
    .config("spark.sql.codegen.wholeStage", "false") // deactivated as the compiled grows to big (exception)
    .master(s"local[1]")
    .getOrCreate()

  import sparkSession.implicits._

  val MODEL_NAME="RegexMatcher"
  //  val MODEL_NAME="TextMatcher"

  val textList = Seq(
    """Quantum computing is the use of quantum-mechanical phenomena such as superposition and entanglement
      |to perform computation. Computers that perform quantum computations are known as quantum computers.
      |Quantum computers are believed to be able to solve certain computational problems, such as integer
      |factorization (which underlies RSA encryption), substantially faster than classical computers.
      |The study of quantum computing is a subfield of quantum information science. Quantum computing
      |began in the early 1980s, when physicist Paul Benioff proposed a quantum mechanical model of the
      |Turing machine. Richard Feynman and Yuri Manin later suggested that a quantum computer had the
      |potential to simulate things that a classical computer could not. In 1994, Peter Shor developed
      |a quantum algorithm for factoring integers that had the potential to decrypt RSA-encrypted
      |communications. Despite ongoing experimental progress since the late 1990s, most researchers
      |believe that "fault-tolerant quantum computing is still a rather distant dream." In recent years,
      |investment into quantum computing research has increased in both the public and private sector.
      |On 23 October 2019, Google AI, in partnership with the U.S. National Aeronautics and Space
      |Administration (NASA), published a paper in which they claimed to have achieved quantum supremacy.
      |While some have disputed this claim, it is still a significant milestone in the history of
      |quantum computing.""".stripMargin,
    """Instacart has raised a new round of financing that makes it one of the most valuable private companies
      |in the U.S., leapfrogging DoorDash, Palantir and Robinhood. Amid surging demand for grocery delivery
      |due to the coronavirus pandemic, Instacart has raised $225 million in a new funding round led by DST
      |Global and General Catalyst. The round increases Instacart’s valuation to $13.7 billion, up from $8
      |billion when it last raised money in 2018. John Doe, followed by leader Quantum test million hundred""".stripMargin)

  val exactMatches = Array("Quantum", "million", "payments", "index", "market share", "gap", "market",
    "measure", "aspects", "accounts", "king" )

  val regexRules = Array("""Quantum\s\w+""", """million\s\w+""", """John\s\w+, followed by leader""",
    """payment.*?\s""", """rall.*?\s""", """\d\d\d\d""", """\d+ Years""")

  val file: File = if (MODEL_NAME.equals("TextMatcher")) {
    val file = new File("textToMatch.txt")
    val bw = new BufferedWriter(new FileWriter(file))
    for (exactMatch <- exactMatches) {
      bw.write(exactMatch)
      bw.write("\n")
    }
    bw.close()
    file
  } else {
    val file = new File("regexToMatch.txt")
    val bw = new BufferedWriter(new FileWriter(file))
    for (regexRule <- regexRules) {
      bw.write(regexRule)
      bw.write("\n")
    }
    bw.close()

    file
  }

  val documentAssembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

  val nlpPipeline = new Pipeline()

  if (MODEL_NAME.equals("TextMatcher")) {
    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")
    val textMatcher = new TextMatcher()
      .setInputCols(Array("document", "token"))
      .setOutputCol("matchedText")
      .setCaseSensitive(false)
      .setEntities("textToMatch.txt", "TEXT")

    nlpPipeline.setStages(Array(documentAssembler, tokenizer, textMatcher))

  } else {
    val regexMatcher = new RegexMatcher()
      .setInputCols("document")
      .setStrategy("MATCH_ALL")
      .setOutputCol("matchedText")
      .setRules("regexToMatch.txt", ",")

    //    val regexMatcherModel = new RegexMatcherModel()
    //      .setInputCols("document")
    //      .setOutputCol("matchedTextWithModel")
    //      .setRules("")

    nlpPipeline.setStages(Array(documentAssembler, regexMatcher))

  }


  val data = textList.toDF("text")

  val pipelineModel: PipelineModel = nlpPipeline.fit(data)

  val nlpData: DataFrame = pipelineModel.transform(data)

  nlpData.select("matchedText").show(false)

  //  val annotated: Map[String, Seq[Annotation]] = new LightPipeline(pipelineModel).fullAnnotate(textList.head)
  //  annotated.foreach(println)

  //  annotated.get("document").head.equals(Annotation("document", 0, 9, "result..."))

  file.delete()
}
