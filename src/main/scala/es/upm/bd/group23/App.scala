package es.upm.bd.group23

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.log4j.{Level, Logger}

/**
 * ● Load the input data, previously stored at a known location.
 * ● Select, process and transform the input variables, to prepare them for training the model.
 * ● Perform some basic analysis of each input variable.
 * ● Create a machine learning model that predicts the arrival delay time.
 * ● Validate the created model and provide some measure of its accuracy.
 */
object App {
  // Input file location
  val inputFilePath = "735926286_T_ONTIME_REPORTING.csv"

  def main(args : Array[String]) {
    /*Logger.getLogger("org").setLevel(Level.WARN)
    val conf = new SparkConf().setAppName("My first Spark application")
    val sc = new SparkContext(conf)
    val data = sc.textFile("file:///tmp/98.txt")
    val numAs = data.filter(line => line.contains("a")).count()
    val numBs = data.filter(line => line.contains("b")).count()
    println(s"Lines with a: ${numAs}, Lines with b: ${numBs}")*/

    // 1. Load the input data, previously stored at a known location.
    val conf = new SparkConf().setAppName("Arrival Delay Predictor Application")
    val sc = new SparkContext(conf)
    val data = sc.textFile(inputFilePath)
    data.foreach(println(_))


  }
}
