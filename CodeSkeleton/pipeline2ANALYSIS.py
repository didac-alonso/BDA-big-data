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


#------------------------------------------------------------------------------------------------------------------------------------------------------#
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
#   7. The model is saved into a folder called "ModelPipeline" in the resources folder.
#       NOTE: The model is also returned, since we couldn't save it in our computers because of an error related with windows I&O.
#------------------------------------------------------------------------------------------------------------------------------------------------------#


def p2Analysis(data):

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

    # We specify the label and features to the decision tree.
    dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol = 'features')
    
    # Chain indexer, assembler and tree in a Pipeline, to use the model.
    # So the indexer and assembler will transform the data before entering the model.
    # NOTE: we don't use the LabelIndexer in the pipeline, since to predict is not needed.
    pipeline = Pipeline(stages=[featureIndexer, assembler, dt])


    trainingData = labelIndexer.transform(trainingData)
    
    # 5. Train model.
    model = pipeline.fit(trainingData)

    testData = labelIndexer.transform(testData)

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

    # 7. Save the model. It is commented since we had problems with the I/O in Windows.
    # model.write().overwrite().save("ModelPipeline")

    return model
        
