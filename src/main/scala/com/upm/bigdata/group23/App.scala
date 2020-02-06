package com.upm.bigdata.group23

/**
 * @author ${user.name}
 */
/*object App {
  
  def foo(x : Array[String]) = x.foldLeft("")((a,b) => a + b)
  
  def main(args : Array[String]) {
    println( "Hello World!" )
    println("concat arguments = " + foo(args))
  }

}*/

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.log4j.{Level, Logger}
object App {
  def main(args : Array[String]) {
    Logger.getLogger("org").setLevel(Level.WARN)
    val conf = new SparkConf().setAppName("My first Spark application")
    val sc = new SparkContext(conf)
    val data = sc.textFile("file:///tmp/98.txt")
    val numAs = data.filter(line => line.contains("a")).count()
    val numBs = data.filter(line => line.contains("b")).count()
    println(s"Lines with a: ${numAs}, Lines with b: ${numBs}")
  }
}
