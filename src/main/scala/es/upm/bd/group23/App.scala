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
  val inputFilePath = "C:\\Users\\TRCECG\\Desktop\\ArrivalDelayPedrictor\\735926286_T_ONTIME_REPORTING.csv"

  def main(args : Array[String]) {

    // 1. Load the input data, previously stored at a known location.
    val conf = new SparkConf().setAppName("Arrival Delay Predictor Application")
    val sc = new SparkContext(conf)
    val data = sc.textFile(inputFilePath)
    // data.foreach(println(_))
    // 2. Select, process and transform the input variables, to prepare them for training the model.
    // 2.1. Gets a RDD with a list with all columns per line and skips header from csv file
    val rdd = data.map(line => {line.split(",")})
      .mapPartitionsWithIndex { (idx, it) => if (idx == 0) it.drop(1) else it }
    rdd.foreach(f=>{
      println("Col1:"+f(0)+",Col2:"+f(1))
    })
    // 2.2. Remove forbidden variables:
    /** 2.1 Remove forbidden variables:
     * ● ArrTime (ARR_TIME)
     * ● ActualElapsedTime (ACTUAL_ELAPSED_TIME)
     * ● AirTime (AIR_TIME)
     * ● TaxiIn (TAXI_IN)
     * ● Diverted (DIVERTED)
     * ● CarrierDelay (CARRIER_DELAY)
     * ● WeatherDelay (WEATHER_DELAY)
     * ● NASDelay (NAS_DELAY)
     * ● SecurityDelay (SECURITY_DELAY)
     * ● LateAircraftDelay (LATE_AIRCRAFT_DELAY)
     *
     */

  }
}
