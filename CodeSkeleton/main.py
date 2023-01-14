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
# ---------------------------------------- #
# PIPELINE 1 FUNCTIONS:
from pipeline1MANAGEMENT import p1Management
# ---------------------------------------- #
# PIPELINE 2 FUNCTIONS:
from pipeline2ANALYSIS import p2Analysis
# ---------------------------------------- #
# PIPELINE 3 FUNCTIONS:
from pipeline3RTClassifier import p3RTClassifier
from pipeline3RTClassifier import Evaluate
# ---------------------------------------- #

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



def interface(spark, AMOS, DW, sc, queryCache):
    print("Welcome! \n Given an aircraftID and a given day a trained model will return a boolean indicating if it is going to have an unexpected operation interruption in the seven following days.")
    print("These are the model metrics:")
    f=open("modelmetrics.txt", "r")
    fl =f.readlines()
    for x in fl:
        print(x)
    while True:
        aircraftID = str(input("Introduce the aircraft ID (AA-AAA):"))
        timeID = str(input("Introduce the day (YYYY-MM-DD):"))
        query = spark.createDataFrame([[aircraftID,timeID]],["aircraftid","timeid"])
        # If it is not empty, then there's only one row and we already have the value stored:
        resultRow = queryCache.join(query, queryCache.aircraftid == query.aircraftid & queryCache.timeid == query.timeid,"leftsemi")
        if len(resultRow) == 0: #this query hasn't already been computed in this session
            resultRow = p3RTClassifier(aircraftID, timeID, spark, AMOS, DW, sc) #find solution
            try:
                queryCache =  queryCache.unionAll(resultRow) #update query cache
            except: # If there is an error, that means that some necessary data is missing and nothing can be done
                return
        printResults(resultRow)


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

    AMOS = (spark.read
        .format("jdbc")
        .option("driver","org.postgresql.Driver")
        .option("url",
        "jdbc:postgresql://postgresfib.fib.upc.edu:6433/AMOS?sslmode=require")
        .option("dbtable", "oldinstance.operationinterruption")
        .option("user", "didac.alonso")
        .option("password", "DB100301")
        .load())
    
    DW = (spark.read
        .format("jdbc")
        .option("driver","org.postgresql.Driver")
        .option("url",
        "jdbc:postgresql://postgresfib.fib.upc.edu:6433/DW?sslmode=require")
        .option("dbtable", "public.aircraftutilization")
        .option("user", "didac.alonso")
        .option("password", "DB100301")
        .load())

    # if not path.exists("DecisionTreeModel.model"):
    DATA, matrixDATA = p1Management(spark,AMOS,DW)
    model = p2Analysis(DATA,sc)


    printResults(Evaluate("XY-LOL", "2012-03-01", spark, DW, model),"XY-LOL", "2012-03-01")
    
    # p3RTClassifier(spark, DW, model)