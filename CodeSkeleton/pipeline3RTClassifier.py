import os
from os import path
import sys
import pyspark
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType,StructField, StringType, DateType, FloatType, BooleanType 
from pyspark.sql.functions import lit
import pyspark.sql.functions as F
from pyspark.sql.window import Window

# HADOOP_HOME = "C:/Users/USER/Desktop/CED/Q5/BDA/PROJECTE2/BDA-big-data/CodeSkeleton/resources/hadoop_home"
# JDBC_JAR = "C:/Users/USER/Desktop/CED/Q5/BDA/PROJECTE2/BDA-big-data/CodeSkeleton/resources/postgresql-42.2.8.jar"
# PYSPARK_PYTHON = "python3"
# PYSPARK_DRIVER_PYTHON = "python3"


HADOOP_HOME = "C:/UNI/Quart/BDA/BDA-big-data/CodeSkeleton/resources/hadoop_home"
JDBC_JAR = "C:/UNI/Quart/BDA/BDA-big-data/CodeSkeleton/resources/postgresql-42.2.8.jar"
PYSPARK_PYTHON = "python3"
PYSPARK_DRIVER_PYTHON = "python3"


def printResults(resultRow,aircraftID,timeID):
    result = resultRow.collect()
    for row in result:
        labelString = "WILL HAVE" if row["prediction"] else "WILL NOT HAVE"
        nextdate = F.to_date(timeID,'yyyy-MM-dd')
        print("==>",aircraftID,labelString,"an unexpected operation interruption in a range of 7 days from",timeID, "(included)")


def interface(spark, DW, queryCache, model):
    print("Welcome! \n Given an aircraftID and a given day a trained model will return a boolean indicating if it is going to have an unexpected operation interruption in the seven following days.")
    print("These are the model metrics:")
    f=open("modelmetrics.txt", "r")
    fl =f.readlines()
    for x in fl:
        print(x)
    while True:
        aircraftID = str(input("Introduce the aircraft ID (AA-AAA):"))
        timeID = str(input("Introduce the day (YYYY-MM-DD):"))
        aircraftID,timeID = "XY-LOL", "2012-03-01"
        query = spark.createDataFrame([[aircraftID,timeID]],["aircraftid","timeid"])
        # If it is not empty, then there's only one row and we already have the value stored:
        resultRow = queryCache.join(query, (queryCache.aircraftid == query.aircraftid) & (queryCache.timeid == query.timeid),"leftsemi")
        if  resultRow.count() == 0: #this query hasn't already been computed in this session
            resultRow = Evaluate(aircraftID, timeID, spark, DW, model) #find solution
            try:
                queryCache =  queryCache.union(resultRow) #update query cache
            except: # If there is an error, that means that some necessary data is missing and nothing can be done
                return
        printResults(resultRow)
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
    # We try to retrieve the average value of the 3453 sensor from the csv files
    try:
        CSVdata = readCSVdata(spark,aircraftID,timeID)
    except:
        print("There is no avaliable information regarding sensor 3453 for aricraft ", aircraftID," at ", timeID)
        print("Due to lacking model features, no answer can be provided")
        return 
    # We try to retrieve the KPIs from the DW 
    try:
        dataFeatures = CSVdata.join(DW, on = ['aircraftid','timeid'], how = 'inner').withColumnRenamed('flighthours','FH')\
        .withColumnRenamed('flightcycles','FC').withColumnRenamed('delayedminutes','DM')
    except:
        print("There is no available KPIs for aricraft ", aircraftID," at ", timeID)
        print("Due to lacking model features, no answer can be provided")
        return 
    print("AAAAAAAA")
    predictions = model.transform(dataFeatures)
    predictions.show()
    print("BBBBBBBB")

    return predictions


    # Falta la part de carregar el model i fer la predicció. En principi la row amb les features ja està calculada.
    # Ha de retornar un df amb l'estructura :    modelSchema = StructType([
    #     StructField('aircraftid', StringType(), True),
    #     StructField('timeid', DateType(), True),
    #     StructField('FH', FloatType(), True),
    #     StructField('FC', FloatType(), True),
    #     StructField('DM', FloatType(), True),
    #     StructField('value', FloatType(), True) --> fins aquí ja està guardat al "dataFeatures"
    #     StructField('label',  BooleanType(), True) --> aquesta és la que s'ha d'afegir depenent del que torni el model (True si hi ha unexpected OI/ False si NO n'hi ha)
    # ])

def p3RTClassifier(spark, DW, model):
    print("aaaaaaaaaaaaaa")
    modelSchema = StructType([
        StructField('aircraftid', StringType(), True),
        StructField('timeid', DateType(), True),
        StructField('FH', FloatType(), True),
        StructField('FC', FloatType(), True),
        StructField('DM', FloatType(), True),
        StructField('value', FloatType(), True),
        StructField('label',  BooleanType(), True)
    ])

    queryCache = spark.createDataFrame([], schema = modelSchema)  
    queryCache = interface(spark, DW, queryCache, model)
    return queryCache