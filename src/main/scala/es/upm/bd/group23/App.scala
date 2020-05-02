package es.upm.bd.group23

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.ml.linalg.{Matrix, Vectors}
import org.apache.spark.ml.regression.{LinearRegression, RandomForestRegressor}
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.ml.stat.Correlation
//import org.apache.spark.sql.catalyst.expressions.{Lag, Window}


/**
 * ● Load the input data, previously stored at a known location.
 * ● Select, process and transform the input variables, to prepare them for training the model.
 * ● Perform some basic analysis of each input variable.
 * ● Create a machine learning model that predicts the arrival delay time.
 * ● Validate the created model and provide some measure of its accuracy.
 */
object App {
  // Input file location
  //val inputFilePath = "./870075876_T_ONTIME_REPORTING.csv"
  val inputFilePath = "./196328912_T_ONTIME_REPORTING.csv"
  val forbiddenVariables = Seq("ARR_TIME", "ACTUAL_ELAPSED_TIME", "AIR_TIME", "TAXI_IN", "DIVERTED", "CARRIER_DELAY",
    "WEATHER_DELAY", "NAS_DELAY", "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY")
  val uselessVariables = Seq("OP_CARRIER_FL_NUM","CRS_DEP_TIME","DEP_TIME","TAXI_OUT", "CANCELLATION_CODE","DISTANCE")
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
      // 2.4 Removes no available values.
      .na.drop()
    // 2.5 Correlation matrix.
    /*val inputMatrix = new VectorAssembler()
      .setInputCols(correlationUsefulVariables)
      .setOutputCol("features")
      .transform(df)
    val Row(coeff: Matrix) = Correlation.corr(inputMatrix, "features", "spearman").head
    println("Spearman correlation matrix :\n" + coeff.toString)
    // 2.6 Group, order and filter.
    val dfSorted = df.orderBy(df.col("YEAR").desc, df.col("MONTH").desc, df.col("DAY_OF_MONTH").desc,
      df.col("DAY_OF_WEEK").desc).filter(df("CANCELLED") === 0)*/

    // 2.7 Creates new variables.
    //, DELAY_FACTOR, DELAY_CUMULATIVE  〖DELAY_FACTOR 〗_(n+1)  =  〖CRS_ELAPSED_TIME〗_(n+1)/〖CRS_ELAPSED_TIME〗_n
    /*val dfFinal = df.withColumn("REAL_ARR_TIME", df.col("DEP_DELAY") +
        df.col("ARR_DELAY") + df.col("CRS_ARR_TIME"))
      .withColumn("DELAY_FACTOR", Lag("DELAY_FACTOR", 1, null).over(window))*/

    // 2.8 Modeling
    // Splits the data into train and test datasets
    val Array(train, test) = df.randomSplit(Array(.8,.2), 42)

    val assembler = new VectorAssembler()
      .setInputCols(correlationUsefulVariables)
      .setOutputCol("features")

    // Modeling:
    // Instantiate the model and fit the training dataset into the model.
    val lr = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol(targetVariable)
      .setMaxIter(10)
      .setElasticNetParam(0.8)

    var randomForestRegressor = new RandomForestRegressor()
      .setFeaturesCol("features")
      .setLabelCol(targetVariable)

    /*println(s"Coefficients: ${lrModel.coefficients}")
    println(s"Intercept: ${lrModel.intercept}")
    val trainingSummary = lrModel.summary

     println(s"numIterations: ${trainingSummary.totalIterations}")
     //println(s"objectiveHistory: ${trainingSummary.objectiveHistory.toList}")
     trainingSummary.residuals.show()
     println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
     println(s"r2: ${trainingSummary.r2}")*/

    // Then user the model on the test data to get the predictions and lastly evaluating the score
    // of the predictions with the evaluation metric.

    val pipeline = new Pipeline()
      .setStages(Array(assembler,lr))
    val lrModel = pipeline.fit(train)

    val pipelineRandomForest = new Pipeline()
      .setStages(Array(assembler,randomForestRegressor))
    val rfModel = pipelineRandomForest.fit(train)

    val prediction = lrModel.transform(test)
    prediction.show(truncate=false)

    val predictionRandomForest = rfModel.transform(test)
    predictionRandomForest.show(truncate=false)

    // 2.9 Evaluating
    /***
     * We can use the RegressionEvaluator to obtain the R2, MSE or RMSE. It required two columns,
     * ARR_DELAY and prediction to evaluate the model.
     */
    // 2.9.1 Evaluate model with area under ROC
    val evaluator = new RegressionEvaluator()
      .setLabelCol(targetVariable)
      .setPredictionCol("prediction")
      .setMetricName("r2")

    // 2.9.2 Measure the accuracy of pipeline model
    val pipelineAccuracy = evaluator.evaluate(prediction)
    println("The model accuracy is: " + pipelineAccuracy)

    val randomForestAccuracy = evaluator.evaluate(predictionRandomForest)
    println("The Random Forest model accuracy is: " + randomForestAccuracy)

    /*dfSorted.write
      .option("header","true")
      .csv("../spark_output")*/

    // 2.1. Gets a RDD with a list with all columns per line and skips header from csv file
    /*val rdd = data.map(line => {line.split(",")})
      // 2.2 Remove forbidden variables:
      .mapPartitionsWithIndex { (idx, it) => if (idx == 0) it.drop(1) else it }

    rdd.foreach(f=>{
      println("Col1:"+f(0)+",Col2:"+f(1))
    })*/

  }
}
