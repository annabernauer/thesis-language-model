package com.mailytica.thesis.language.model.util

object Utility {

  val DELIMITER : String = "%&§§&%"                       //important: no char that needs to be escaped in regex

  def printToFile(f: java.io.File)(op: java.io.PrintWriter => Unit) {
    val p = new java.io.PrintWriter(f)
    try { op(p) } finally { p.close() }
  }

}
