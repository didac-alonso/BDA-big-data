import os
import sys
import pyspark
from pyspark import SparkConf
from pyspark.sql import SparkSession

import MR_Test

#Path to the resources folder of the project, use the full path on Windows
HADOOP_HOME = "./resources/hadoop_home"
#Use the full path on Windows for PYSPARK_PYTHON and PYSPARK_DRIVER_PYTHON, for instance: r"C:\SOFT\Python3\python.exe"
PYSPARK_PYTHON = "python3"
PYSPARK_DRIVER_PYTHON = "python3"

if(__name__== "__main__"):
    os.environ["HADOOP_HOME"] = HADOOP_HOME
    sys.path.append(HADOOP_HOME + "\\bin")
    os.environ["PYSPARK_PYTHON"] = PYSPARK_PYTHON
    os.environ["PYSPARK_DRIVER_PYTHON"] = PYSPARK_DRIVER_PYTHON

    conf = SparkConf()  # create the configuration

    spark = SparkSession.builder \
        .config(conf=conf) \
        .master("local") \
        .appName("MapReduce") \
        .getOrCreate()

    sc = pyspark.SparkContext.getOrCreate()

    mr = sc.textFile("resources/hamlet.txt").flatMap(lambda t: MR_Test.map(1,t)).groupByKey().flatMap(lambda t: MR_Test.reduce(t[0],t[1]))

    for x in mr.collect():
        print(x)