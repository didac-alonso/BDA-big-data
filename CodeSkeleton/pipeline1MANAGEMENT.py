import os
import sys
import pyspark
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType,StructField, StringType, DateType, FloatType
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


# This pipeline reads the data from the sensors, aircraft utilization and operation interruption, and creates a dataframe with the following columns:
# aircraftid, timeid, FH, FC, DM, value, label
# where aircraftid is the aircraft id, timeid is the day of the measurement, FH is flighthours, FC is flightcycles, DM is delayedminutes,
# value is the average values for the sensor at some day, label is a binary variable that indicates if the aircraft is gonna interrupt
# inexpectedly its operations or not we put a 1 if the aircraft is not in a scheduled mantenaince and 0 if it is
# The pipeline consists in the following steps:
#   1. Read the data from the sensors, and calculate the mean of the values for each day
#   2. Join the data from the sensors with the KPIs from aircraft utilization
#   3. Filter the data to only include information about subsystem 3453
#   4. Join the data from the sensors with the operation interruption table
#       NOTE: The data is filtered to only include the interruptions that are not scheduled, and then using a full join 
#       we add the unscheduledOI column to the data. If the interruption is not scheduled, the value is 1, and 0 otherwise
#   5. Check wether the following 7 days have an unscheduled OI or not. If not, the label column is set to 0
#   6. Transform the DataFrame to matrix and save it as a .csv
#       NOTE: At the end it returns the DataFrame since the model needs it to train the model, so it is not necessary to read it again


# Reads the csv files from resources/trainingData folder and returns a list of DafaFrames which also contains
# the last 6 characters from the filename without the csv extension
def readSensorsData(spark):

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

        # We create a new column with the i-following day and the i-following unscheduledOI value, this is the next i record
        DATA = DATA.withColumn('following_uOI', F.lead('unscheduledOI', offset= i).over(w)).withColumn('following_day', F.lead('timeid', offset = i).over(w))

        # We assign 1 to the label column if:
        # - If the time between the i-following day and the current timeID is up to 7 days and there is an unscheduledOI 
        #       in that day that is, unscheduledOI == 1
        # - If there is an unscheduledOI in the current day
        # - In a previous iteration label == 1 has been assigned
        DATA = DATA.withColumn('label', F.when(((F.col('following_uOI') == 1)&(F.date_add(F.col('timeid'),7) > F.col('following_day')))\
                    |(F.col('unscheduledOI') == 1)|(F.col('label') == 1),1).otherwise(0))

    DATA = DATA.select('aircraftid','timeid','FH','FC','DM','value','label')
    return DATA


# def check_following_days(DATA):
#     """"This function checks if the following 7 days are scheduled or not. If they are not, the maintenance column is set to 0."""

#     # We need to order the data by aircraftid and timeid, to be able to use the lead function
#     w = Window.partitionBy('aircraftid').orderBy('timeid')

#     for i in range(1,8):

#         # We create a new column with the i-following day and the i-following scheduled value, this is the next i row
#         DATA = DATA.withColumn('following_sch', F.lead('Scheduled', offset= i).over(w)).withColumn('following_day', F.lead('timeid', offset = i).over(w))

#         # We check if the i-following day is scheduled or not, if it is not, we set the maintenance column to 0
#         DATA = DATA.withColumn('maintenance', F.when(((F.col('following_sch') == 0)&(F.date_add(F.col('timeid'),7) > F.col('following_day')))|(F.col('maintenance') == 0)\
#                     |(F.col('Scheduled') == 0),0).otherwise(1)).sort('aircraftid','timeid').select('aircraftid','timeid','FH','FC','DM','value','maintenance', 'Scheduled')

#     return DATA




def DFtoMatrix(df):

    return None

def p1Management(spark,AMOS,DW):
    
    # 1. We read the data from the sensors and calculate the mean for each day
    files = readSensorsData(spark)
    
    # 2. We join the data from the sensors with the KPIs from aircraft utilization
    # We also rename the KPÌs column names to its acronyms
    KPI_SENSOR = files.join(DW, on = ['aircraftid','timeid'], how = 'inner').withColumnRenamed('flighthours','FH')\
        .withColumnRenamed('flightcycles','FC').withColumnRenamed('delayedminutes','DM')

    # We select the columns that we need from the operation interruption table
    amosDATA = AMOS.select(F.col('aircraftregistration').alias('aircraftid'), F.col('subsystem').alias('subsystem'), F.col('starttime').alias('timeid'), F.col('kind').alias('kind'))

    # 3. We filter the data to keep only the subsystem 3453 and the unscheduled operation interruptions. We drop the columns we do not use any more (kind and subsystem)
    amosDATA = amosDATA.filter(F.col('subsystem') == 3453).withColumn('Scheduled', F.when((F.col('kind') == 'Maintenance')|(F.col('kind') == 'Revision'), 1).otherwise(0)).drop('kind','subsystem')

    # DATA = DATA.filter((F.col('subsystem') == 3453)&((F.col('kind') == "AircraftOnGround")|(F.col('kind') == "Delay")|(F.col('kind') == "Safety"))).drop('kind','subsystem')
    
    # We format the timeid column by year-month-day to be able to join it with the data from the sensors.
    amosDATA = amosDATA.withColumn('timeid', F.to_date(F.col('timeid'),'yyyy-MM-dd'))

    # We add attribute unscheduledOI, which is 1 for days where an unscheduledOI happens. At this moment, all AMOS rows encode days of this type
    amosDATA = amosDATA.withColumn('unscheduledOI', lit(1))

    # 4. We join the data from operation interruption with the data from the sensors+KPIs, we perform a full join to keep all the data from the sensors, and add 0 
    # for the (aircraft,time) that don't have any interruption (in the "unscheduledOI"). 
    # We also need to keep all the DATA from operation interruption, because we need to check if there is an unscheduled operation interruption
    # during the 7 following days independenlty of if there is a sensor value or not.
    DATA = amosDATA.join(KPI_SENSOR, on = ['aircraftid','timeid'], how = 'full')\
                .select('aircraftid','timeid','unscheduledOI','FH','FC','DM','value').fillna(0, subset=['unscheduledOI'])

    # DATA = amosDATA.join(KPI_SENSOR, on = ['aircraftid','timeid'], how = 'full')\
    #             .select('aircraftid','timeid','Scheduled','FH','FC','DM','value').fillna(1, subset=['Scheduled']).withColumn('maintenance', F.when(F.col('Scheduled') == 0, 0).otherwise(1))\
    
    # We add the label column initialized at 0,
    # it will be assigned in the fucntion check following_days --> 
    # label = 1 --> unscheduled OI in the next 7 days
    # label = 0 --> NO unscheduled OI in the next 7 days
    DATA = DATA.withColumn('label',lit(0))

    # 5. We check if there is an unscheduled operation interruption in the following 7 days
    DATA = check_following_days(DATA)

    # Now we drop the rows that have null values at 'value' column, because we can't train the model with them
    #   NOTE: We can perform this step since we only filled nas in the unscheduledOI column, so it identifies
    #       the days without data from the sensors
    DATA = DATA.na.drop(subset = ['value'])
        
    # 6. We transform the dataframe to a matrix and save it in a csv_file
    matrixDATA = DFtoMatrix(DATA)

    return DATA, matrixDATA
