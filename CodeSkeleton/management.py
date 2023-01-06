import os
import sys
import pyspark
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType,StructField, StringType, DateType, FloatType
from pyspark.sql.functions import lit
import pyspark.sql.functions as F
from pyspark.sql.window import Window

HADOOP_HOME = "C:/UNI/Quart/BDA/BDA-big-data/CodeSkeleton/resources/hadoop_home"
JDBC_JAR = "C:/UNI/Quart/BDA/BDA-big-data/CodeSkeleton/resources/postgresql-42.2.8.jar"
PYSPARK_PYTHON = "python3"
PYSPARK_DRIVER_PYTHON = "python3"

# Reads the csv files from resources/trainingData folder and returns a list of DafaFrames which also contains 
# the last 6 characters from the filename without the csv extension
def readTrainingData(spark):
    
    # We create a empty dataframe to store the data
    schema = StructType([
        StructField('aircraftid', StringType(), True),
        StructField('timeid', DateType(), True),
        StructField('value', FloatType(), True)
    ])
    
    # The DataFrame has the following schema: aircraft, date, value
    # Where aircraft is the aircraft id, date is the day of the measurement and value is the average values for the sensor
    data = spark.createDataFrame([], schema = schema)  
    
    
    
    # for that iterates beyond the filenames of the files in the folder resources/trainingData and creates a DataFrame with the content of each file
    for filename in os.listdir("resources/trainingData"):
        if filename.endswith(".csv"):

            file = spark.read.csv("resources/trainingData/" + filename, sep = ';' ,header=True, inferSchema=True)

            # transform "date" column from datetime.datetime to datetime.date
            file = file.withColumn("timeid", F.to_date(F.col("date"), "yyyy-MM-dd"))\
                .withColumn("aircraftid", lit(filename[-10:-4])).select("aircraftid","timeid","value")
            
            # finally we add the value to the dataframe
            data = data.union(file)
    data = data.groupBy(["timeid",'aircraftid']).agg(F.mean("value").alias('value'))

    return data
    

def check_following_days(DATA):
    """"This function checks if the following 7 days are scheduled or not. If they are not, the maintenance column is set to 0."""

    # We need to order the data by aircraftid and timeid, to be able to use the lead function   
    w = Window.partitionBy('aircraftid').orderBy('timeid')

    for i in range(1,8):
        
        # We create a new column with the i-following day and the i-following scheduled value, this is the next i row
        DATA = DATA.withColumn('following_sch', F.lead('Scheduled', offset= i).over(w)).withColumn('following_day', F.lead('timeid', offset = i).over(w))
        
        # We check if the i-following day is scheduled or not, if it is not, we set the maintenance column to 0
        DATA = DATA.withColumn('maintenance', F.when(((F.col('following_sch') == 0)&(F.date_add(F.col('timeid'),7) > F.col('following_day')))|(F.col('maintenance') == 0)\
                    |(F.col('Scheduled') == 0),0).otherwise(1)).sort('aircraftid','timeid').select('aircraftid','timeid','FH','FC','DM','value','maintenance', 'Scheduled')
    
    return DATA


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
    
    
    files = readTrainingData(spark)
    
    # We join the data from the sensors with the KPIs
    KPI_SENSOR = files.join(DW, on = ['aircraftid','timeid'], how = 'inner').withColumnRenamed('flighthours','FH')\
        .withColumnRenamed('flightcycles','FC').withColumnRenamed('delayedminutes','DM')
    
    # We select the columns that we need from the operation interruption table
    DATA = AMOS.select(F.col('aircraftregistration').alias('aircraftid'), F.col('subsystem').alias('subsystem'), F.col('starttime').alias('timeid'), F.col('kind').alias('kind'))
    
    # We filter the data to keep only the subsystem 3453, and we create a new column with 1 if the interruption is scheduled, and 0 if it is not, we drop the columns that we don't need anymore
    DATA = DATA.filter(F.col('subsystem') == 3453).withColumn('Scheduled', F.when((F.col('kind') == 'Maintenance')|(F.col('kind') == 'Revision'), 1).otherwise(0)).drop('kind','subsystem')
    
    # We format the timeid column by year-month-day to be able to join it with the data from the sensors
    DATA = DATA.withColumn('timeid', F.to_date(F.col('timeid'),'yyyy-MM-dd'))
    
    # We join the data from operation interruption with the data from the sensors+KPIs, we perform a full join to keep all the data from the sensors, and add 1('no mantainance')
    # for the aircrafts that don't have any interruption. We also need to keep all the DATA from operation interruption, because we need to check if the following 7 days are scheduled or not
    DATA = DATA.join(KPI_SENSOR, on = ['aircraftid','timeid'], how = 'full')\
                .select('aircraftid','timeid','Scheduled','FH','FC','DM','value').withColumn('maintenance', F.when(F.col('Scheduled') == 0, 0).otherwise(1))\
                
    DATA = check_following_days(DATA)
    
    # Now we drop the rows that have null values at 'value' column, because we can't train the model with them
    DATA = DATA.na.drop(subset = ['value'])
    
    DATA.show(30)