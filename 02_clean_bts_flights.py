from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Initialize Spark
spark = SparkSession.builder \
    .appName("BTS Flight Cleaning") \
    .getOrCreate()

print("Loading raw Parquet data...")
df_flights = spark.read.parquet("hdfs://namenode:9000/data/processed/flights_raw")

raw_count = df_flights.count()
print("Raw record count: {:,}".format(raw_count))

print("Starting cleaning process...")

# 1. Remove cancelled flights
df_clean = df_flights.filter(F.col("Cancelled") == 0)

# 2. Handle missing delays (Fill with 0)
df_clean = df_clean.fillna({"DepDelay": 0.0, "ArrDelay": 0.0})

# 3. Remove outlier delays (keep between -60 and 500 mins)
df_clean = df_clean.filter((F.col("DepDelay") >= -60) & (F.col("DepDelay") <= 500))

# 4. Create proper Timestamp string for joining (padding hours like '930' to '0930')
df_clean = df_clean.withColumn(
    "CRSDepTimeStr",
    F.lpad(F.col("CRSDepTime").cast("string"), 4, "0")
)

# Convert to actual Timestamp type
df_clean = df_clean.withColumn(
    "DepTimestamp",
    F.to_timestamp(F.concat_ws(" ", F.col("FlightDate"), F.col("CRSDepTimeStr")), "yyyy-MM-dd HHmm")
).drop("CRSDepTimeStr")

# 5. Create Severe Delay flag (for your classification model later)
df_clean = df_clean.withColumn(
    "SevereDelay",
    F.when(F.col("DepDelay") >= 30, 1).otherwise(0)
)

clean_count = df_clean.count()
print("Clean record count: {:,}".format(clean_count))
print("Total rows removed: {:,}".format(raw_count - clean_count))

print("Writing cleaned data to partitioned Parquet files...")

# Save cleaned data, partitioned by Year and Month for faster querying later
df_clean.write \
    .mode("overwrite") \
    .partitionBy("Year", "Month") \
    .parquet("hdfs://namenode:9000/data/processed/flights_cleaned")

print("Data cleaning complete!")