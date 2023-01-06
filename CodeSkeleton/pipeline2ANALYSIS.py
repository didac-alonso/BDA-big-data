import os
import sys
import pyspark
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pipeline1MANAGEMENT import DATA # GUARDEM EL SDF RESULTANT DE LA PIPELINE DE MANAGEMENT COM A VARIABLE
from pyspark.mllib.evaluation import MulticlassMetrics
# https://people.apache.org/~pwendell/spark-nightly/spark-master-docs/latest/mllib-evaluation-metrics.html#binary-classification

HADOOP_HOME = "C:/Users/USER/Desktop/CED/Q5/BDA/PROJECTE2/resources/hadoop_home"
JDBC_JAR = "C:/Users/USER/Desktop/CED/Q5/BDA/PROJECTE2/BDA-big-data/CodeSkeleton/resources/postgresql-42.2.8.jar"
PYSPARK_PYTHON = "python3"
PYSPARK_DRIVER_PYTHON = "python3"




if(__name__== "__main__"):
    os.environ["HADOOP_HOME"] = HADOOP_HOME
    sys.path.append(HADOOP_HOME + "\\bin")
    os.environ["PYSPARK_PYTHON"] = PYSPARK_PYTHON
    os.environ["PYSPARK_DRIVER_PYTHON"] = PYSPARK_DRIVER_PYTHON

    #conf = SparkConf()  # create the configuration
    #conf.set("spark.jars", JDBC_JAR)

    spark = SparkSession.builder \
        .master("local") \
        .appName("Training") \
        .config("spark.jars.packages", "org.postgresql:postgresql:42.2.12") \
        .getOrCreate()
    sc = pyspark.SparkContext.getOrCreate()

    data = DATA.withColumnRenamed('maintenance','label')

    # Index labels, adding metadata to the label column.
    # Fit on whole dataset to include all labels in index.
    labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)
    # Automatically identify categorical features, and index them.
    # We specify maxCategories so features with > 4 distinct values are treated as continuous.

    features = ['aircraftid','timeid','FH','FC','DM','value']
    featureIndexer = VectorIndexer(inputCol=features, outputCol="indexedFeatures").fit(data)

    # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = data.randomSplit([0.7, 0.3])

    # Train a DecisionTree model.
    dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures")

    # Chain indexers and tree in a Pipeline
    pipeline = Pipeline(stages=[labelIndexer, featureIndexer, dt])

    # Train model.  This also runs the indexers.
    model = pipeline.fit(trainingData)

    # Make predictions.
    predictions = model.transform(testData)

    # Select example rows to display.
    predictions.select("prediction", "indexedLabel", "features").show(5)

    predictionAndLabels = testData.map(lambda lp: (float(model.predict(lp.features)), lp.label))
    metrics = MulticlassMetrics(predictionAndLabels)
    recall = metrics.recall()
    print("Recall = %s" % recall)
    # Select (prediction, true label) and compute test error
    evaluator = MulticlassClassificationEvaluator( #he canviat pel multiclass
        labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Test Error = %g " % (1.0 - accuracy))

    treeModel = model.stages[2]
    # summary only
    print(treeModel)
    
    #Create and point to your pipelines here
