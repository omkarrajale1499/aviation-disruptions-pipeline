"""
Reads NOAA ISD-Lite files from HDFS. Format: 13 columns, space-separated, no header.
-9999 = missing. Raw values scaled by 10 for temp, dew point, pressure, wind, precip.
Writes staging Parquet partitioned by year+month.
"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, IntegerType, StringType

HDFS = "hdfs://namenode:9000"
INPUT = f"{HDFS}/user/team/flight_delay/raw/weather/*/*"  # year/file
OUTPUT = f"{HDFS}/user/team/flight_delay/staging/weather"

spark = (SparkSession.builder
         .appName("ingest_weather")
         .config("spark.sql.shuffle.partitions", "8")
         .config("spark.speculation", "false")
         .config("spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version", "1")
         .config("spark.hadoop.mapreduce.fileoutputcommitter.marksuccessfuljobs", "true")
         .getOrCreate())

# ISD-Lite is fixed-width-ish but easier to parse as whitespace-separated
# Schema: year, month, day, hour, air_temp, dew_point, slp, wind_dir,
#         wind_speed, sky_cond, precip_1h, precip_6h
schema = StructType([
    StructField("year",        IntegerType()),
    StructField("month",       IntegerType()),
    StructField("day",         IntegerType()),
    StructField("hour_utc",    IntegerType()),
    StructField("air_temp",    IntegerType()),
    StructField("dew_point",   IntegerType()),
    StructField("slp",         IntegerType()),
    StructField("wind_dir",    IntegerType()),
    StructField("wind_speed",  IntegerType()),
    StructField("sky_cond",    IntegerType()),
    StructField("precip_1h",   IntegerType()),
    StructField("precip_6h",   IntegerType()),
])

df = (spark.read
      .option("header", "false")
      .option("delimiter", " ")
      .option("ignoreLeadingWhiteSpace", "true")
      .option("multiline", "false")
      .schema(schema)
      .csv(INPUT))

# Extract station_id from input file path
df = df.withColumn("filepath", F.input_file_name())
df = df.withColumn(
    "station_id",
    F.regexp_extract(F.col("filepath"), r"(\d{6}-\d{5})-\d{4}", 1)
)
df = df.drop("filepath")

# Replace -9999 with null and scale
def clean_scaled(col_name, scale=10.0):
    return F.when(F.col(col_name) == -9999, None).otherwise(F.col(col_name) / scale)

df = df.select(
    "station_id",
    "year", "month", "day", "hour_utc",
    clean_scaled("air_temp").alias("air_temp_c"),
    clean_scaled("dew_point").alias("dew_point_c"),
    clean_scaled("slp").alias("slp_hpa"),
    F.when(F.col("wind_dir") == -9999, None).otherwise(F.col("wind_dir")).alias("wind_dir"),
    clean_scaled("wind_speed").alias("wind_speed_ms"),
    F.when(F.col("sky_cond") == -9999, None).otherwise(F.col("sky_cond")).alias("sky_cond"),
    clean_scaled("precip_1h").alias("precip_1h_mm"),
    clean_scaled("precip_6h").alias("precip_6h_mm"),
)

# Build observation timestamp in UTC
df = df.withColumn(
    "obs_ts_utc",
    F.to_timestamp(
        F.format_string("%04d-%02d-%02d %02d:00:00",
                        "year", "month", "day", "hour_utc"),
        "yyyy-MM-dd HH:mm:ss"
    )
)

print(f"Weather rows: {df.count():,}")

(df.repartition(8, "year", "month")
   .write
   .mode("overwrite")
   .partitionBy("year", "month")
   .parquet(OUTPUT))

spark.stop()
