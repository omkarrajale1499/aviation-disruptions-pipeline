from pyspark.sql import SparkSession
import pyspark.sql.functions as F

spark = SparkSession.builder.appName("EDA_Aggregations").getOrCreate()

print("Loading Processed Data Lake...")
df_flights = spark.read.parquet("hdfs://namenode:9000/data/processed/flights_cleaned")
df_weather = spark.read.parquet("hdfs://namenode:9000/data/processed/weather_parsed")

# 1. Top 20 Busiest Airports vs Average Delay
print("Aggregating Airport Delays...")
airport_delays = df_flights.groupBy("Origin") \
    .agg(F.count("*").alias("Total_Flights"), F.avg("DepDelay").alias("Avg_Delay")) \
    .orderBy(F.desc("Total_Flights")).limit(20)

# 2. Average Delay by Hour of Day
# CRS_DEP_TIME is formatted like 1530 for 3:30 PM. Floor divide by 100 gives us the hour (15)
print("Aggregating Hourly Delays...")
hourly_delays = df_flights.withColumn("Hour", F.floor(F.col("CRSDepTime") / 100)) \
    .groupBy("Hour") \
    .agg(F.avg("DepDelay").alias("Avg_Delay"), F.count("*").alias("Total_Flights")) \
    .orderBy("Hour")

# 3. Weather & Delays (Daily National Average)
print("Aggregating Daily Weather and Delays...")
daily_flights = df_flights.groupBy("FlightDate").agg(F.avg("DepDelay").alias("Avg_Flight_Delay"))

daily_weather = df_weather.groupBy("WeatherDate").agg(
    F.avg("AirTemp_Celsius").alias("Avg_Temp_C"),
    F.avg("WindSpeed_ms").alias("Avg_Wind_ms")
)

# Save outputs to HDFS instead of the local container file system
print("Saving summary tables to HDFS...")
airport_delays.coalesce(1).write.mode("overwrite").option("header", "true").csv("hdfs://namenode:9000/data/processed/eda_airport_delays")
hourly_delays.coalesce(1).write.mode("overwrite").option("header", "true").csv("hdfs://namenode:9000/data/processed/eda_hourly_delays")
daily_flights.coalesce(1).write.mode("overwrite").option("header", "true").csv("hdfs://namenode:9000/data/processed/eda_daily_flights")
daily_weather.coalesce(1).write.mode("overwrite").option("header", "true").csv("hdfs://namenode:9000/data/processed/eda_daily_weather")

print("Aggregations safely stored in HDFS!")