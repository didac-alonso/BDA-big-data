import pyspark
from pyspark.sql import SparkSession

sc = pyspark.SparkContext.getOrCreate()
sess = SparkSession(sc)
pyspark.sql.SparkSession.conf("spark.jars.packages", "org.postgresql:postgresql:42.2.12")

JDBC = (sess.read
.format("jdbc")
.option("driver","org.postgresql.Driver")
.option("url",
"jdbc:postgresql://postgresfib.fib.upc.edu:6433/AMOS?sslmode=require")
.option("dbtable", "public.operationinterruption")
.option("user", "DBdidac.alonso")
.option("password", "DB100301")
.load())
