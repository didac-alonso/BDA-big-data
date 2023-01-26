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


HADOOP_HOME = "C:/UNI/Quart/BDA/BDA-big-data/CodeSkeleton/resources/hadoop_home"
JDBC_JAR = "C:/UNI/Quart/BDA/BDA-big-data/CodeSkeleton/resources/postgresql-42.2.8.jar"
PYSPARK_PYTHON = "python3"
PYSPARK_DRIVER_PYTHON = "python3"



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
    DATA = p1Management(spark,AMOS,DW)
    model = p2Analysis(DATA)

    
    p3RTClassifier(spark, DW, model)