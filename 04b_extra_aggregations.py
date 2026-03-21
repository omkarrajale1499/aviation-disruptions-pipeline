from pyspark.sql import SparkSession
import pyspark.sql.functions as F

spark = SparkSession.builder.appName("EDA_Extra").getOrCreate()

print("Loading flights data...")
df_flights = spark.read.parquet("hdfs://namenode:9000/data/processed/flights_cleaned")

# 1. Airline Performance
print("Aggregating Airlines...")
airline_delays = df_flights.groupBy("Reporting_Airline") \
    .agg(F.avg("DepDelay").alias("Avg_Delay"), F.count("*").alias("Total_Flights")) \
    .filter(F.col("Total_Flights") > 5000) # Filter out tiny regional carriers

# 2. Temporal Heatmap Data (Day of Week vs Hour)
print("Aggregating Heatmap Matrix...")
heatmap_data = df_flights.withColumn("Hour", F.floor(F.col("CRSDepTime") / 100)) \
    .groupBy("DayOfWeek", "Hour") \
    .agg(F.avg("DepDelay").alias("Avg_Delay"))

print("Saving to HDFS...")
airline_delays.coalesce(1).write.mode("overwrite").option("header", "true").csv("hdfs://namenode:9000/data/processed/eda_airline_delays")
heatmap_data.coalesce(1).write.mode("overwrite").option("header", "true").csv("hdfs://namenode:9000/data/processed/eda_heatmap_data")

print("Extra Aggregations Complete!")