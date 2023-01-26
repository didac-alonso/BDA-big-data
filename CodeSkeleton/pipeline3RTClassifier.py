import os
from os import path
import sys
import pyspark
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.types import StructType,StructField, StringType, DateType, FloatType, IntegerType 
from pyspark.sql.functions import lit
import pyspark.sql.functions as F
from pyspark.sql.window import Window

#------------------------------------------------------------------------------------------------------------------------------------------------------#
# This pipeline implements the run-time classifier. That is, when the user enters manually
# a series of queries (aircraftID,timeid), each record is enriched with the features necessary for the model to make a prediction.
# In other words, from the csv files (just the ones that matter) the average value for the 3453 sensor is extracted, and the KPIs from the DW.
# For that reasons, the only (aircraftid,timeid) pairs supported by this implementation are those which have both enrichments.
# If that is the case, then the model makes the prediction and outputs it. In case there are repeated queries, a queryCache is kept 
# (and returned at the end to consult all queries)
# The pipeline consists in the following steps:
#   0. Create the empty queryCache
#   1. Read the queries  from the console
#   2. Check that a concrete query has not already been computed
#   3a. If it is not the case then look for the average value for sensor 3453 for that aircraft and day in the TrainingData
#   4a. Retrieve the KPIs (FH,FC,DM) for that (aircraftID,timeID) from the DW
#   5a. Merge that information to generate the entries for the model to predict
#   6a. Make a prediction using the model
#   3b. Retrieve the result from the queryCache
#   4b and 7a. Output the prediction result 
#------------------------------------------------------------------------------------------------------------------------------------------------------#


# STEP 4b and 7a --> Output the prediction result 
def printResults(resultRow,aircraftID,timeID):
    """Recieves a pyspark dataframe and prints the 'prediction' result"""
    result = resultRow.collect()
    for row in result:
        labelString = "WILL HAVE" if row["label"] else "WILL NOT HAVE"
        print("==>",aircraftID,labelString,"an unexpected operation interruption in a range of 7 days from",timeID, "(that day being included)")


def interface(spark, DW, queryCache, model):
    print("Welcome! \n Given an aircraftID and a given day a trained model will return a boolean indicating if it is going to have an unexpected operation interruption in the seven following days.")
    print("These are the model metrics:")
    f=open("modelmetrics.txt", "r")
    fl =f.readlines()
    for x in fl:
        print(x)
    # STEP 1 --> Read the queries from a console
    inp = input("Enter 'EXIT' to exit, 'CACHE' to display all queries done and anything else to use the Run-Time Classifier: ")
    while inp!="EXIT":
        if inp =="CACHE":
            queryCache.show()
        else:
            aircraftID = str(input("Introduce the aircraft ID (AA-AAA):"))
            timeID = str(input("Introduce the day (YYYY-MM-DD):"))
            query = spark.createDataFrame([[aircraftID,timeID]],["aircraftid","timeid"])
            # STEP 2 --> Check that that concrete query has not already been computed
            # STEP 3b. Retrieve the result from the queryCache --> If it is not empty, then there's only one row and we already have the value stored:
            resultRow = queryCache.join(query, (queryCache.aircraftid == query.aircraftid) & (queryCache.timeid == query.timeid),"leftsemi")
            if  resultRow.count() == 0: #this query hasn't already been computed in this session
                resultRow = Evaluate(aircraftID, timeID, spark, DW, model) #find solution
                # In case the entry is untreatable --> NO RESULT ROW FOUND
                if resultRow != None:
                    resultRow = resultRow.withColumnRenamed('prediction','label').select('aircraftid','timeid','FH','FC','DM','value','label')
                    queryCache =  queryCache.union(resultRow) #update query cache
            if resultRow != None:
                printResults(resultRow,aircraftID,timeID)
        inp = input("Press intro to continue, enter 'EXIT' to exit, 'CACHE' to display all queries done and anything else to use the Run-Time Classifier:  ")
    return queryCache

def readCSVdata(spark, aircraftID, timeID):
    # We create a empty dataframe to store the data
    schema = StructType([
        StructField('aircraftid', StringType(), True),
        StructField('timeid', DateType(), True),
        StructField('value', FloatType(), True)
    ])
    data = spark.createDataFrame([], schema = schema)  
    adaptedtimeid = timeID[8:] + timeID[5:7] + timeID[2:4] # This is the date in the format that the csv files follow
    # We aggregate the data in all csv files regarding that aircraft, that day
    for filename in os.listdir("resources/trainingData"):
        if filename.startswith(adaptedtimeid) and filename.endswith(aircraftID+".csv"):
            file = spark.read.csv("resources/trainingData/" + filename, sep = ';' ,header=True, inferSchema=True)
            # transform "date" column from datetime.datetime to datetime.date
            file = file.withColumn("timeid", F.to_date(F.col("date"), "yyyy-MM-dd"))\
                .withColumn("aircraftid", lit(filename[-10:-4])).select("aircraftid","timeid","value")
            # finally we add the value to the dataframe
            data = data.union(file)
    data = data.groupBy(["timeid",'aircraftid']).agg(F.mean("value").alias('value'))
    return data


def Evaluate(aircraftID, timeID, spark, DW, model):
    #STEP 3a --> If it is not the case then look for the average value for sensor 3453 for that aircraft and day in the TrainingData
    # We try to retrieve the average value of the 3453 sensor from the csv files
    try:
        CSVdata = readCSVdata(spark,aircraftID,timeID)
        assert CSVdata.count() > 0 
    except:
        print("There is no avaliable information regarding sensor 3453 for aricraft ", aircraftID," at ", timeID)
        print("Due to lacking model features, no answer can be provided")
        return 
    # We try to retrieve the KPIs from the DW 
    try:
        #STEP 4a --> Retrieve the KPIs (FH,FC,DM) for that (aircraftID,timeID) from the DW
        #STEP 5a --> Merge that information to generate the entries for the model to predict
        dataFeatures = CSVdata.join(DW, on = ['aircraftid','timeid'], how = 'inner').withColumnRenamed('flighthours','FH')\
        .withColumnRenamed('flightcycles','FC').withColumnRenamed('delayedminutes','DM')
        assert dataFeatures.count() > 0
    except:
        print("There is no available KPIs for aricraft ", aircraftID," at ", timeID)
        print("Due to lacking model features, no answer can be provided")
        return 
    #STEP 6a --> Make a prediction using the model
    predictions = model.transform(dataFeatures)
    return predictions


def p3RTClassifier(spark, DW, model):
    # STEP 0 --> Create the empty queryCache
    modelSchema = StructType([
        StructField('aircraftid', StringType(), True),
        StructField('timeid', DateType(), True),
        StructField('FH', FloatType(), True),
        StructField('FC', FloatType(), True),
        StructField('DM', FloatType(), True),
        StructField('value', FloatType(), True),
        StructField('label',  IntegerType(), True)
    ])
    # If we had no problems saving the model, we would not use it as a function parameter and we would execute the following line:
    # model = PipelineModel.read().load("ModelPipeline")
    queryCache = spark.createDataFrame([], schema = modelSchema)  
    queryCache = interface(spark, DW, queryCache, model)
    print("These are all the queries done:")
    queryCache.show()