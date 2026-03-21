from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType, StringType

# 1. Initialize Spark Session
spark = SparkSession.builder \
    .appName("BTS Flight Ingestion") \
    .getOrCreate()

# 2. Define schema matching the EXACT raw CSV headers and order
raw_schema = StructType([
    StructField("YEAR", IntegerType(), True),
    StructField("QUARTER", IntegerType(), True),
    StructField("MONTH", IntegerType(), True),
    StructField("DAY_OF_MONTH", IntegerType(), True),
    StructField("DAY_OF_WEEK", IntegerType(), True),
    StructField("FL_DATE", StringType(), True),
    StructField("OP_UNIQUE_CARRIER", StringType(), True),
    StructField("TAIL_NUM", StringType(), True),
    StructField("OP_CARRIER_FL_NUM", StringType(), True),
    StructField("ORIGIN", StringType(), True),
    StructField("ORIGIN_CITY_NAME", StringType(), True),
    StructField("DEST", StringType(), True),
    StructField("DEST_CITY_NAME", StringType(), True),
    StructField("CRS_DEP_TIME", IntegerType(), True),
    StructField("DEP_TIME", IntegerType(), True),
    StructField("DEP_DELAY", DoubleType(), True),
    StructField("ARR_DELAY", DoubleType(), True),
    StructField("CANCELLED", DoubleType(), True),
    StructField("DIVERTED", DoubleType(), True),
    StructField("CRS_ELAPSED_TIME", DoubleType(), True),
    StructField("ACTUAL_ELAPSED_TIME", DoubleType(), True),
    StructField("DISTANCE", DoubleType(), True)
])

print("Reading raw CSV files from HDFS...")

# 3. Read the data with the raw schema
df_flights_raw = spark.read \
    .option("header", "true") \
    .schema(raw_schema) \
    .csv("hdfs://namenode:9000/data/raw/bts_flights/*/*.csv")

# 4. Rename columns to match our clean Midterm Plan schema
df_clean_names = df_flights_raw.toDF(
    "Year", "Quarter", "Month", "DayofMonth", "DayOfWeek", "FlightDate",
    "Reporting_Airline", "Tail_Number", "Flight_Number_Reporting_Airline",
    "Origin", "OriginCityName", "Dest", "DestCityName",
    "CRSDepTime", "DepTime", "DepDelay", "ArrDelay",
    "Cancelled", "Diverted", "CRSElapsedTime", "ActualElapsedTime", "Distance"
)

# 5. Count the rows
total_records = df_clean_names.count()
print("\n========================================")
print("Total raw flight records: {:,}".format(total_records))
print("========================================\n")
df_clean_names.printSchema()

print("Writing data to Parquet format...")

# 6. Save as a single, optimized Parquet table
df_clean_names.write \
    .mode("overwrite") \
    .parquet("hdfs://namenode:9000/data/processed/flights_raw")

print("Ingestion complete!")