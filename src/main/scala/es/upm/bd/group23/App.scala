package es.upm.bd.group23

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.regression.{LinearRegression, RandomForestRegressor}


/**
 * ● Load the input data, previously stored at a known location.
 * ● Select, process and transform the input variables, to prepare them for training the model.
 * ● Perform some basic analysis of each input variable.
 * ● Create a machine learning model that predicts the arrival delay time.
 * ● Validate the created model and provide some measure of its accuracy.
 */
object App {
  // Input file location
  val inputFilePath = "../196328912_T_ONTIME_REPORTING.csv"

  val forbiddenVariables = Seq("ARR_TIME", "ACTUAL_ELAPSED_TIME", "AIR_TIME", "TAXI_IN", "DIVERTED", "CARRIER_DELAY",
    "WEATHER_DELAY", "NAS_DELAY", "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY")
  val uselessVariables = Seq("OP_CARRIER_FL_NUM","CRS_DEP_TIME","DEP_TIME","TAXI_OUT", "CANCELLATION_CODE","DISTANCE")
  val correlationUsefulVariables: Array[String] = Array("DEP_DELAY","CRS_ARR_TIME","ARR_DELAY","CRS_ELAPSED_TIME")
  val targetVariable = "ARR_DELAY"

  def main(args : Array[String]) {

    // 1. Load the input data, previously stored at a known location.
    val spark = SparkSession
      .builder()
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

    // 3. Modeling
    // 3.1. Splits the data into train and test datasets
    val Array(train, test) = df.randomSplit(Array(.8,.2), 42)

    // 3.2. Create Assembler
    val assembler = new VectorAssembler()
      .setInputCols(correlationUsefulVariables)
      .setOutputCol("features")

    // 3.3. Instantiate the estimators and fit the training dataset into the model.
    val linearRegression = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol(targetVariable)
      .setMaxIter(10)
      .setElasticNetParam(0.8)

    val randomForestRegressor = new RandomForestRegressor()
      .setFeaturesCol("features")
      .setLabelCol(targetVariable)

    // 3.4. Create pipelines and execute them to fit the models
    val linearRegressionPipeline = new Pipeline()
      .setStages(Array(assembler,linearRegression))
    val lrModel = linearRegressionPipeline.fit(train)

    val randomForestPipeline = new Pipeline()
      .setStages(Array(assembler,randomForestRegressor))
    val rfModel = randomForestPipeline.fit(train)

    // 4. Evaluating
    // 4.1. Use the model on the test data to get the predictions
    val linearRegressionPrediction = lrModel.transform(test)
    linearRegressionPrediction.show(truncate=false)

    val randomForestPrediction = rfModel.transform(test)
    randomForestPrediction.show(truncate=false)

    // 4.2. Define the R2 Regression Evaluator
    val evaluator = new RegressionEvaluator()
      .setLabelCol(targetVariable)
      .setPredictionCol("prediction")
      .setMetricName("r2")

    // 4.3. Measure the accuracy of each model
    val pipelineAccuracy = evaluator.evaluate(linearRegressionPrediction)
    println("The Linear Regression model accuracy is: " + pipelineAccuracy)

    val randomForestAccuracy = evaluator.evaluate(randomForestPrediction)
    println("The Random Forest Regression model accuracy is: " + randomForestAccuracy)
  }
}
