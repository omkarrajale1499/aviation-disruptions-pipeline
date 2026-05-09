"""
Reads BTS flight CSVs from HDFS, normalizes schema, filters cancelled/diverted,
writes staging Parquet partitioned by year+month.
Columns verified against actual BTS download format.
"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

HDFS = "hdfs://namenode:9000"
INPUT = f"{HDFS}/user/team/flight_delay/raw/flights/*.csv"
OUTPUT = f"{HDFS}/user/team/flight_delay/staging/flights"

spark = (SparkSession.builder
         .appName("ingest_flights")
         .config("spark.sql.shuffle.partitions", "32")
         .getOrCreate())

df = (spark.read
      .option("header", "true")
      .option("inferSchema", "false")
      .csv(INPUT))

print("Actual columns in CSV:", df.columns)

# Map actual BTS column names → our internal names
df = df.select(
    F.col("YEAR").cast("int").alias("year"),
    F.col("MONTH").cast("int").alias("month"),
    F.col("DAY_OF_MONTH").cast("int").alias("day"),        # was DayofMonth — FIXED
    F.col("DAY_OF_WEEK").cast("int").alias("dow"),         # was DayOfWeek — FIXED
    F.col("FL_DATE").alias("flight_date"),                 # was FlightDate — FIXED
    F.col("OP_UNIQUE_CARRIER").alias("carrier"),           # was Reporting_Airline — FIXED
    F.col("TAIL_NUM").alias("tail"),                       # was Tail_Number — FIXED
    F.col("ORIGIN").alias("origin"),
    F.col("ORIGIN_STATE_ABR").alias("origin_state"),       # was OriginState — FIXED
    F.col("DEST").alias("dest"),
    F.col("DEST_STATE_ABR").alias("dest_state"),           # was DestState — FIXED
    F.col("CRS_DEP_TIME").cast("int").alias("crs_dep_time"),
    F.col("DEP_DELAY").cast("double").alias("dep_delay"),
    F.col("DEP_DEL15").cast("double").alias("dep_del15"),
    F.col("CRS_ARR_TIME").cast("int").alias("crs_arr_time"),
    F.col("ARR_DELAY").cast("double").alias("arr_delay"),
    F.col("CANCELLED").cast("double").alias("cancelled"),
    F.col("DIVERTED").cast("double").alias("diverted"),
    F.col("CRS_ELAPSED_TIME").cast("double").alias("crs_elapsed"),  # was CRSElapsedTime — FIXED
    F.col("DISTANCE").cast("double").alias("distance"),
)

# Filter cancelled/diverted
df = df.filter((F.col("cancelled") == 0) & (F.col("diverted") == 0))
df = df.filter(F.col("dep_delay").isNotNull())
df = df.drop("cancelled", "diverted")

# Build scheduled departure timestamp
df = df.withColumn("crs_dep_hhmm", F.lpad(F.col("crs_dep_time").cast("string"), 4, "0"))
df = df.withColumn("crs_dep_hour", F.substring(F.col("crs_dep_hhmm"), 1, 2).cast("int"))
df = df.withColumn("crs_dep_minute", F.substring(F.col("crs_dep_hhmm"), 3, 2).cast("int"))

df = df.withColumn("flight_date_clean",
    F.to_date(F.col("flight_date"), "M/d/yyyy h:mm:ss a"))


df = df.withColumn(
    "crs_dep_local",
    F.to_timestamp(
        F.concat(
            F.date_format(F.col("flight_date_clean"), "yyyy-MM-dd"),
            F.lit(" "),
            F.lpad(F.col("crs_dep_hour").cast("string"), 2, "0"),
            F.lit(":"),
            F.lpad(F.col("crs_dep_minute").cast("string"), 2, "0"),
            F.lit(":00")
        ),
        "yyyy-MM-dd HH:mm:ss"
    )
)
df = df.withColumn("crs_dep_hour_local", F.date_trunc("hour", F.col("crs_dep_local")))

count = df.count()
print(f"Row count after filtering: {count:,}")
print(f"Sample dep_delay stats:")
df.select(
    F.mean("dep_delay").alias("mean"),
    F.min("dep_delay").alias("min"),
    F.max("dep_delay").alias("max"),
    F.mean("dep_del15").alias("delay_rate")
).show()

(df.repartition(16, "year", "month")
   .write
   .mode("overwrite")
   .partitionBy("year", "month")
   .parquet(OUTPUT))

print(f"Written to {OUTPUT}")
spark.stop()