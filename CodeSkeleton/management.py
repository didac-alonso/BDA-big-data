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

    print(data.show(2))
    return data
    



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
    
    DATA = AMOS.select('aircraftregistration', 'subsystem', 'starttime', 'kind').filter(F.col('subsystem') == 3453)\
        .withColumn('Scheduled', F.when((F.col('kind') == 'Maintenance')|(F.col('kind') == 'Revision'), 0).otherwise(1)).drop('kind','subsystem')\
            .withColumnRenamed('aircraftregistration','aircraftid').withColumnRenamed('starttime','timeid').withColumn('timeid', F.to_date(F.col('timeid'),'yyyy-MM-dd'))\
                .join(files, on = ['aircraftid','timeid'], how = 'inner').join(DW, on = ['aircraftid', 'timeid'], how = 'inner').withColumnRenamed('flighthours','FH')\
                    .withColumnRenamed('flightcycles','FC').withColumnRenamed('delayedminutes','DM').select('aircraftid','timeid','Scheduled','FH','FC','DM','value')
    
    
    
    # We need to put a 0 if there's a 0 in the Scheduled column in the next 7 days for the same aircraft and a 1 if not
    # DATA = DATA.withColumn('following_sch',F.lead('Scheduled', 7).over(Window.partitionBy('aircraftid').orderBy('timeid'))).show(10)
    
    DATA.sort('scheduled','aircraftid','timeid').show(20)    


    w = Window.partitionBy('aircraftid').orderBy('timeid')
    
    DATA = DATA.withColumn('sch', F.when(F.col('scheduled') == 1, 0).otherwise(1)).withColumn('sch', F.lag('sch').over(w)).withColumn('sch', F.lead('sch').over(w))\
        .withColumn('sch', F.when(F.date_add(F.col('timeid'),7) < F.col('timeid'), 0).otherwise(1)).sort('scheduled','aircraftid','timeid').show(20)

                
    # DW_lb = DW_lb.select('aircraft_registration','date','kind').withColumn('kind', F.when(F.col('kind') == 'manteniment', 'unscheduled maintenance' or \
    #     F.col('kind') == 'revisio').otherwise('no maintenance'))
    
    # # Faig right, i poso 0 en els nulls per si no hi ha el KPI calculat, l'assumim com a 0
    # DW_aircraft = DW_aircraft.select('aircraft_registration','date','FH', 'FC', 'DM').withColumnRenamed('aircraft_registration','aircraft') \
    #     .join(DW_lb, on = ['aircraft','date']).na.fill(value = 'no maintenance').join(files, on = ['aircraft','date'], how = 'right').na.fill(value = 0).show(10)
    #     # Faltaria fer el column renamed però per la data però el postgres no va, així que no sé :D
    
            
    # print(type(DW))
    
    # a = DW.select('*')
    # print(a.take(1))
    #Create and point to your pipelines here
