package com.mailytica.thesis.language.model.ngram.cosineSimilarity.annotators

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.sql.types.{ArrayType, DataTypes, StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset}

import java.util.UUID

class ExplodedTransformer(override val uid: String = UUID.randomUUID().toString) extends Transformer {

  val outputCol: Param[String] = new Param[String](this, "outputCol", "output column name")

  def setOutputCol(value: String): ExplodedTransformer = set(outputCol, value)

  def getOutputCol: String = $(outputCol)

  val inputCol: Param[String] = new Param[String](this, "inputCol", "input column name")

  def setInputCol(value: String): ExplodedTransformer = set(inputCol, value)

  def getInputCol: String = $(inputCol)

  setDefault(
    inputCol -> "sentence",
    outputCol -> "document"
  )


  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {

    val actualDataType = schema($(inputCol)).dataType
    require(actualDataType.equals(ArrayType(StringType)), s"Column ${$(inputCol)} must be ArrayType but was actually $actualDataType.")

    schema.add(StructField($(outputCol), DataTypes.StringType, true))
  }

  override def transform(dataset: Dataset[_]): DataFrame = {

    import org.apache.spark.sql.functions._

    dataset.select(explode(col($(inputCol))).as($(outputCol)))
  }

}
