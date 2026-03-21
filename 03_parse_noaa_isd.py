from pyspark.sql import SparkSession
import pyspark.sql.functions as F

spark = SparkSession.builder.appName("NOAA Weather Parsing").getOrCreate()

print("Reading 43.5GB of raw NOAA text files...")
# Read the compressed .gz files as a single string column named 'value'
df_raw = spark.read.text("hdfs://namenode:9000/data/raw/noaa_isd/*/*.gz")

print("Extracting fixed-width features...")
# PySpark substring is 1-indexed: substring(column, start_pos, length)
df_weather = df_raw.select(
    F.substring("value", 5, 6).alias("USAF_Station"),
    F.substring("value", 11, 5).alias("WBAN_Station"),
    F.substring("value", 16, 8).alias("WeatherDate"),
    F.substring("value", 24, 4).alias("WeatherTime"),
    F.substring("value", 66, 4).cast("float").alias("WindSpeed_Scaled"),
    F.substring("value", 88, 5).cast("float").alias("AirTemp_Scaled")
)

# NOAA fills missing data with 9999. Also, temps and wind are scaled by 10.
print("Cleaning missing values and scaling...")
df_clean_weather = df_weather \
    .withColumn("AirTemp_Celsius", 
                F.when(F.col("AirTemp_Scaled") == 9999, None)
                .otherwise(F.col("AirTemp_Scaled") / 10.0)) \
    .withColumn("WindSpeed_ms", 
                F.when(F.col("WindSpeed_Scaled") == 9999, None)
                .otherwise(F.col("WindSpeed_Scaled") / 10.0)) \
    .drop("AirTemp_Scaled", "WindSpeed_Scaled")

print("Writing parsed weather data to Parquet (This will take a while)...")
# Save as highly compressed Parquet
df_clean_weather.write \
    .mode("overwrite") \
    .parquet("hdfs://namenode:9000/data/processed/weather_parsed")

print("Weather Parsing Complete!")