import org.apache.spark._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.classification.BinaryLogisticRegressionSummary
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorAssembler
//import sqlContext.implicits._
//import sqlContext._
import org.apache.spark.sql.functions._
//import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics


    
 
// define the Cancer Observation Schema
case class Obs(clas: Double, thickness: Double, size: Double, shape: Double, madh: Double, epsize: Double, bnuc: Double, bchrom: Double, nNuc: Double, mit: Double)


// function to create a Obs class from an Array of Double.Class Malignant 4 is changed to 1
def parseObs(line: Array[Double]): Obs = {
    Obs(
      if (line(9) == 4.0) 1 else 0, line(0), line(1), line(2), line(3), line(4), line(5), line(6), line(7), line(8)
    )
}
// function to transform an RDD of Strings into an RDD of Double, filter lines with ?, remove first column
def parseRDD(rdd: RDD[String]): RDD[Array[Double]] = {
    rdd.map(_.split(",")).filter(_(6) != "?").map(_.drop(1)).map(_.map(_.toDouble))
}


// load the data into a DataFrame
//val rdd = sc.textFile("data/wbcd.csv")
val rdd = sc.textFile("data/wbcd.csv")

val obsRDD = parseRDD(rdd).map(parseObs)
val obsDF = obsRDD.toDF().cache()
obsDF.registerTempTable("obs")

// Return the schema of this DataFrame
obsDF.printSchema

// Display the top 20 rows of DataFrame 
obsDF.show


//  describe computes statistics for thickness column, including count, mean, stddev, min, and max
obsDF.describe("thickness").show

val sqlContext= new org.apache.spark.sql.SQLContext(sc)
import sqlContext.implicits._


// compute the avg thickness, size, shape grouped by clas (malignant or not) 
sqlContext.sql("SELECT clas, avg(thickness) as avgthickness, avg(size) as avgsize, avg(shape) as avgshape FROM obs GROUP BY clas ").show

// compute avg thickness grouped by clas (malignant or not) 
obsDF.groupBy("clas").avg("thickness").show
//define the feature columns to put in the feature vector
val featureCols = Array("thickness", "size", "shape", "madh", "epsize", "bnuc", "bchrom", "nNuc", "mit")

//set the input and output column names
val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
//return a dataframe with all of the  feature columns in  a vector column
val df2 = assembler.transform(obsDF)
// the transform method produced a new column: features.
df2.show
//  Create a label column with the StringIndexer  
val labelIndexer = new StringIndexer().setInputCol("clas").setOutputCol("label")
val df3 = labelIndexer.fit(df2).transform(df2)
// the  transform method produced a new column: label.
df3.show

//  split the dataframe into training and test data
val splitSeed = 5043 
val Array(trainingData, testData) = df3.randomSplit(Array(0.7, 0.3), splitSeed)

// create the classifier,  set parameters for training
val lr = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)
//  use logistic regression to train (fit) the model with the training data
val model = lr.fit(trainingData)    

// Print the coefficients and intercept for logistic regression
println(s"Coefficients: ${model.coefficients} Intercept: ${model.intercept}")

// run the  model on test features to get predictions
val predictions = model.transform(testData) 
//As you can see, the previous model transform produced a new columns: rawPrediction, probablity and prediction.
predictions.show

//A common metric used for logistic regression is area under the ROC curve (AUC). We can use the BinaryClasssificationEvaluator to obtain the AUC 
// create an Evaluator for binary classification, which expects two input columns: rawPrediction and label.
val evaluator = new BinaryClassificationEvaluator().setLabelCol("label").setRawPredictionCol("rawPrediction").setMetricName("areaUnderROC")
// Evaluates predictions and returns a scalar metric areaUnderROC(larger is better). 
val accuracy = evaluator.evaluate(predictions) 

// Calculate Metrics
val lp = predictions.select( "label", "prediction")
val counttotal = predictions.count()
val correct = lp.filter($"label" === $"prediction").count()
val wrong = lp.filter(not($"label" === $"prediction")).count()
val truep = lp.filter($"prediction" === 0.0).filter($"label" === $"prediction").count()
val falseN = lp.filter($"prediction" === 0.0).filter(not($"label" === $"prediction")).count()
val falseP = lp.filter($"prediction" === 1.0).filter(not($"label" === $"prediction")).count()
val ratioWrong=wrong.toDouble/counttotal.toDouble
val ratioCorrect=correct.toDouble/counttotal.toDouble

// use MLlib to evaluate, convert DF to RDD
val  predictionAndLabels =predictions.select("rawPrediction", "label").rdd.map(x => (x(0).asInstanceOf[DenseVector](1), x(1).asInstanceOf[Double]))
val metrics = new BinaryClassificationMetrics(predictionAndLabels) 
println("area under the precision-recall curve: " + metrics.areaUnderPR)
println("area under the receiver operating characteristic (ROC) curve : " + metrics.areaUnderROC)
// A Precision-Recall curve plots (precision, recall) points for different threshold values, while a receiver operating characteristic, or ROC, curve plots (recall, false positive rate) points. The closer  the area Under ROC is to 1, the better the model is making predictions



