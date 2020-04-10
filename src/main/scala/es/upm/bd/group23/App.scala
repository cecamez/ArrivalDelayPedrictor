package es.upm.bd.group23

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.ml.linalg.{Matrix, Vectors}
import org.apache.spark.ml.stat.Correlation


/**
 * ● Load the input data, previously stored at a known location.
 * ● Select, process and transform the input variables, to prepare them for training the model.
 * ● Perform some basic analysis of each input variable.
 * ● Create a machine learning model that predicts the arrival delay time.
 * ● Validate the created model and provide some measure of its accuracy.
 */
object App {
  // Input file location
  val inputFilePath = "..\\196328912_T_ONTIME_REPORTING.csv"
  val forbiddenVariables = Seq("ARR_TIME", "ACTUAL_ELAPSED_TIME", "AIR_TIME", "TAXI_IN", "DIVERTED", "CARRIER_DELAY",
    "WEATHER_DELAY", "NAS_DELAY", "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY")
  val uselessVariables = Seq("OP_CARRIER_FL_NUM","CRS_DEP_TIME","DEP_TIME","TAXI_OUT", "CANCELLATION_CODE","DISTANCE")
  val categoricalVariables = Seq("YEAR","MONTH", "DAY_OF_MONTH", "DAY_OF_WEEK", "OP_UNIQUE_CARRIER", "TAIL_NUM",
    "ORIGIN","DEST","CANCELLED")
  val correlationUsefulVariables = Array("DEP_DELAY","CRS_ARR_TIME","ARR_DELAY","CRS_ELAPSED_TIME")
  val targetVariable = "ARR_DELAY"

  def main(args : Array[String]) {

    // 1. Load the input data, previously stored at a known location.
    val spark = SparkSession
      .builder()
      //.master("local[1]")
      .appName("Arrival Delay Predictor")
      .getOrCreate();

    // 2. Select, process and transform the input variables, to prepare them for training the model.
    val df = spark.read
      .options(Map("inferSchema"->"true","sep"->",","header"->"true"))
      // 2.1 Load the input data, previously stored at a known location with SparkSession.
      .csv(inputFilePath)
      // 2.2 Removes forbidden variables.
      .drop(forbiddenVariables:_*)
      // 2.3 Removes useless variables.
      .drop(uselessVariables:_*)
      // 2.4 Removes no available values
      .na.drop()
    // 2.5 Correlation matrix of all variables.
    val inputMatrix = new VectorAssembler()
      .setInputCols(correlationUsefulVariables)
      .setOutputCol("features")
      .transform(df)
    val Row(coeff: Matrix) = Correlation.corr(inputMatrix, "features", "spearman").head

    println("Spearman correlation matrix :\n" + coeff.toString)

    df.write
      .option("header","true")
      .csv("../spark_output")

    // 2.1. Gets a RDD with a list with all columns per line and skips header from csv file
    /*val rdd = data.map(line => {line.split(",")})
      // 2.2 Remove forbidden variables:
      .mapPartitionsWithIndex { (idx, it) => if (idx == 0) it.drop(1) else it }

    rdd.foreach(f=>{
      println("Col1:"+f(0)+",Col2:"+f(1))
    })*/

  }
}
