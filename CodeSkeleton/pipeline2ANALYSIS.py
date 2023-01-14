import os
import sys
import pyspark
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer, VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
 # GUARDEM EL SDF RESULTANT DE LA PIPELINE DE MANAGEMENT COM A VARIABLE
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql.functions import when, col

# https://people.apache.org/~pwendell/spark-nightly/spark-master-docs/latest/mllib-evaluation-metrics.html#binary-classification

# HADOOP_HOME = "C:/Users/USER/Desktop/CED/Q5/BDA/PROJECTE2/BDA-big-data/CodeSkeleton/resources/hadoop_home"
# JDBC_JAR = "C:/Users/USER/Desktop/CED/Q5/BDA/PROJECTE2/BDA-big-data/CodeSkeleton/resources/postgresql-42.2.8.jar"
# PYSPARK_PYTHON = "python3"
# PYSPARK_DRIVER_PYTHON = "python3"


HADOOP_HOME = "C:/UNI/Quart/BDA/BDA-big-data/CodeSkeleton/resources/hadoop_home"
JDBC_JAR = "C:/UNI/Quart/BDA/BDA-big-data/CodeSkeleton/resources/postgresql-42.2.8.jar"
PYSPARK_PYTHON = "python3"
PYSPARK_DRIVER_PYTHON = "python3"


# This pipeline is used to train the model, which is a decision tree, to predict whether the aircraft will need an unscheduled maintenance within the next 7 days or not.
# The input is the dataframe that contains the features and the label (maintenance) and the output is the trained model. Which is also saved into a folder called "model" in the resources folder,
# ready to be used for the third pipeline.

# The pipeline is consists in the following steps:
#   1. The label is indexed, so it can be used in the model.
#   2. aircraftid is indexed, so it can be used in the model.
#   3. The features are assembled into a vector. To be used in the model.
#       The features are: aircraftid, timeid, FH, FC, DM and value.
#   4. The data is partitioned into training and test sets.       
#   5. The model is trained using the DecisionTreeClassifier.
#   6. The model is tested using the test set. The metrics used are accuracy and recall.
#       NOTE: The results of the test are saved into a .txt file
#   7. The model is saved into a folder called "model" in the resources folder.
#       NOTE: The model is also returned, since we couldn't save it in our computers because of an error related with windows I&O.



if(__name__== "__main__"):
    os.environ["HADOOP_HOME"] = HADOOP_HOME
    sys.path.append(HADOOP_HOME + "\\bin")
    os.environ["PYSPARK_PYTHON"] = PYSPARK_PYTHON
    os.environ["PYSPARK_DRIVER_PYTHON"] = PYSPARK_DRIVER_PYTHON


    spark = SparkSession.builder \
        .master("local") \
        .appName("Training") \
        .config("spark.jars.packages", "org.postgresql:postgresql:42.2.12") \
        .getOrCreate()
    sc = pyspark.SparkContext.getOrCreate()


# Since the data is unbalanced, we need to calculate the weights of each class.
# The weights are inversely proportional to the frequency of each class.
# def calculate_weights(data):

#     total_instances = data.count()

#     # Firstly we compute the frequency of each class
#     class_counts = data.groupBy("label").count().collect()
    
#     class_frequencies = {row["label"]: row["count"]/total_instances for row in class_counts}

#     # We calculate the weight as the inverse of the frequency
#     weights = {label: total_instances/(freq*2) for label, freq in class_frequencies.items()}

#     # We add it to the dataframe
#     data = data.withColumn("weight", when(col("label").isin(weights.keys()), col("label").cast("double")).otherwise(1.0))    
    
#     return data



def p2Analysis(data):
    # data = DATA.withColumnRenamed('maintenance','label')

    # 1. Index labels, to be used in the model.
    labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)

    # 2. Index aircraftid, to be used in the model.
    featureIndexer = StringIndexer(inputCol='aircraftid', outputCol="indexedFeatures").fit(data)

    # We specify the non-categorical features
    non_categorical_features = ['FH','FC','DM','value']

    # 3. Assemble features into a vector.
    assembler = VectorAssembler(inputCols=["indexedFeatures", *non_categorical_features], outputCol="features")

    # 4. Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = data.randomSplit([0.7, 0.3])

    # trainingData = calculate_weights(trainingData)

    # We specify the label and features to the decision tree.
    dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol = 'features')
    
    # Chain indexers, assembler and tree in a Pipeline, to use the model.
    # So the indexers and assembler will transform the data before entering the model.
    pipeline = Pipeline(stages=[labelIndexer,featureIndexer, assembler, dt])

    # 5. Train model.
    model = pipeline.fit(trainingData)

    # We make predictions on the test set. To evaluate the model.
    predictions = model.transform(testData)

    # Select the rows from predictions.
    predictions = predictions.select("prediction", "indexedLabel", "features")

    # 6. Evaluate the model.
    # First we calculate the accuracy
    evaluator = MulticlassClassificationEvaluator(
        labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
    
    accuracy = evaluator.evaluate(predictions)

    # Now we calculate the recalls
    RECevaluator = MulticlassClassificationEvaluator(
        labelCol="indexedLabel", predictionCol="prediction", metricName = 'recallByLabel', metricLabel = 0.0
    )
    
    recall0 = RECevaluator.evaluate(predictions)
    
    RECevaluator = MulticlassClassificationEvaluator(
        labelCol="indexedLabel", predictionCol="prediction", metricName = 'recallByLabel', metricLabel = 1.0
    )

    recall1 = RECevaluator.evaluate(predictions)



    print('Test Error = %g' % (1.0 - accuracy))
    f= open("modelmetrics.txt","w+")
    f.write("------------------------------------------------------------\n")
    f.write("DECISION TREE MODEL\n")
    f.write("LABEL --> Unexpected Operation Interruption in the next following days from timeID for aircraftID\n")
    f.write("FEATURES: \n x1 --> aircraftID \
                        \n x2 --> timeID \
                         \n x3 --> FH (flight hours that day) \
                          \n x4 --> FC (flight cycles that day) \
                           \n x5 --> DM (minutes delayed that day) \
                            \n x6 --> value (Average measurement for sensor 3453)") 
    f.write("\nModel Accuracy --> "+ str(100*accuracy)+"%"+'\n')
    f.write("Model Recall label 0 --> " + str(100*recall0)+"%"+'\n')
    f.write("Model Recall label 1 --> " + str(100*recall1)+"%"+'\n')
    f.write("------------------------------------------------------------")
    f.close()   

    # model.save("model")
    # model.write().overwrite().save("C:/UNI/Quart/BDA/Joel_11-1_falta(pip3)/DecisionTreeModel.model")
    # model.write().overwrite().save("model_pipeline")

    return model
        
