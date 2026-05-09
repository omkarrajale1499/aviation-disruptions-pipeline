"""
1. Builds airport (IATA) <-> station (USAF-WBAN) crosswalk via ICAO.
2. Joins flights to weather on (origin, scheduled_hour_utc).
   Note: We treat flight local time as UTC for simplicity. This is wrong by
   3-8 hours per airport but acceptable for a 1.5-day project — the weather
   conditions persist over multi-hour windows so the bias is bounded.
   Document this limitation in the report.
3. Engineers features.
4. Writes train (2023) / test (2024) Parquet.
"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import math

HDFS = "hdfs://namenode:9000"
FLIGHTS = f"{HDFS}/user/team/flight_delay/staging/flights"
WEATHER = f"{HDFS}/user/team/flight_delay/staging/weather"
ISD_HISTORY = f"{HDFS}/user/team/flight_delay/raw/isd-history-fresh.csv"
AIRPORTS = f"{HDFS}/user/team/flight_delay/raw/airports.csv"
OUTPUT_TRAIN = f"{HDFS}/user/team/flight_delay/processed/train"
OUTPUT_TEST = f"{HDFS}/user/team/flight_delay/processed/test"

spark = (SparkSession.builder
         .appName("join_and_features")
         .config("spark.sql.shuffle.partitions", "64")
         .config("spark.sql.autoBroadcastJoinThreshold", "10485760")  # 10MB
         .getOrCreate())

# ---------- 1. Build crosswalk (proximity-based, verified stations only) ----------
isd_raw = spark.read.option("header", "true").csv(ISD_HISTORY)

# Get station IDs that actually exist in weather staging
weather_station_ids = (spark.read.parquet(WEATHER)
    .select("station_id").distinct())

# Build station lookup with coordinates — no CTRY filter, just use verified stations
isd_all = (isd_raw
    .filter(isd_raw.LAT.isNotNull())
    .filter(isd_raw.LON.isNotNull())
    .filter(isd_raw.LAT != "")
    .filter(isd_raw.LON != "")
    .select(
        F.concat_ws("-", F.trim(isd_raw.USAF),
                   F.trim(isd_raw.WBAN)).alias("station_id"),
        F.col("LAT").cast("double").alias("s_lat"),
        F.col("LON").cast("double").alias("s_lon")
    ))

isd_us = isd_all.join(weather_station_ids, "station_id", "inner")
print("Verified weather stations with coords:", isd_us.count())

# Load airports
airports_raw = spark.read.option("header", "true").csv(AIRPORTS)
airports_cw = (airports_raw
    .filter("iso_country = 'US'")
    .filter("iata_code is not null")
    .filter("iata_code != ''")
    .filter("type in ('large_airport','medium_airport')")
    .select(
        F.trim(F.col("iata_code")).alias("iata"),
        F.col("latitude_deg").cast("double").alias("a_lat"),
        F.col("longitude_deg").cast("double").alias("a_lon")
    ))

# Proximity join — nearest verified station per airport
crossed = airports_cw.crossJoin(F.broadcast(isd_us))
crossed = crossed.withColumn("dist",
    F.sqrt(F.pow(F.col("a_lat") - F.col("s_lat"), 2) +
           F.pow(F.col("a_lon") - F.col("s_lon"), 2)))

w = Window.partitionBy("iata").orderBy("dist")
crosswalk = (crossed
    .withColumn("rn", F.row_number().over(w))
    .filter("rn = 1")
    .filter("dist < 1.0")   # slightly wider threshold
    .select("iata", "station_id"))

crosswalk.cache()
print("Crosswalk size:", crosswalk.count())
crosswalk.show(20, truncate=False)
# ---------- 2. Load flights and join crosswalk ----------
flights = spark.read.parquet(FLIGHTS)

# Attach origin station
flights = (flights.alias("f")
           .join(crosswalk.alias("cw"),
                 F.col("f.origin") == F.col("cw.iata"), 
                 "left")
           .select("f.*", F.col("cw.station_id").alias("origin_station")))

# Drop flights with no station match
flights = flights.filter(F.col("origin_station").isNotNull())

# Round local timestamp to hour as join key (treating local as UTC — see docstring)
flights = flights.withColumn(
    "weather_join_hour",
    F.date_trunc("hour", F.col("crs_dep_local"))
)

# ---------- 3. Load weather ----------
weather = (spark.read.parquet(WEATHER)
           .select(
               "station_id",
               F.col("obs_ts_utc").alias("weather_join_hour"),
               "air_temp_c", "dew_point_c", "slp_hpa",
               "wind_speed_ms", "sky_cond", "precip_1h_mm"
           ))

# ---------- 4. Join ----------
# Both keys are exact-match (station_id + hour timestamp)
joined = flights.join(
    weather,
    (flights.origin_station == weather.station_id) &
    (flights.weather_join_hour == weather.weather_join_hour),
    "left"
).drop(weather.station_id).drop(weather.weather_join_hour)

# ---------- 5. Feature engineering ----------
# Cyclical time features
joined = (joined
    .withColumn("hour_sin", F.sin(F.col("crs_dep_hour") * (2 * math.pi / 24)))
    .withColumn("hour_cos", F.cos(F.col("crs_dep_hour") * (2 * math.pi / 24)))
    .withColumn("month_sin", F.sin(F.col("month") * (2 * math.pi / 12)))
    .withColumn("month_cos", F.cos(F.col("month") * (2 * math.pi / 12)))
    .withColumn("dow_sin", F.sin(F.col("dow") * (2 * math.pi / 7)))
    .withColumn("dow_cos", F.cos(F.col("dow") * (2 * math.pi / 7)))
)

# Calendar binary
joined = (joined
    .withColumn("is_weekend", F.col("dow").isin([6, 7]).cast("int"))
    .withColumn("is_peak_hour",
                F.col("crs_dep_hour").isin([7, 8, 17, 18, 19]).cast("int"))
    .withColumn("season",
                F.when(F.col("month").isin([12, 1, 2]), "winter")
                 .when(F.col("month").isin([3, 4, 5]), "spring")
                 .when(F.col("month").isin([6, 7, 8]), "summer")
                 .otherwise("fall"))
)

# Weather severity (defensive — handle nulls)
joined = (joined
    .withColumn("is_freezing",
                F.when(F.col("air_temp_c") < 0, 1).otherwise(0))
    .withColumn("is_overcast",
                F.when(F.col("sky_cond") >= 7, 1).otherwise(0))
    .withColumn("is_strong_wind",
                F.when(F.col("wind_speed_ms") > 10, 1).otherwise(0))
    .withColumn("is_precip",
                F.when(F.col("precip_1h_mm") > 0.1, 1).otherwise(0))
    .withColumn("weather_severity",
                F.col("is_freezing") + F.col("is_overcast") +
                F.col("is_strong_wind") + F.col("is_precip"))
)

# Relative humidity (Magnus formula approximation)
joined = joined.withColumn(
    "rel_humidity",
    F.when((F.col("air_temp_c").isNotNull()) & (F.col("dew_point_c").isNotNull()),
           100 * F.exp((17.625 * F.col("dew_point_c")) /
                       (243.04 + F.col("dew_point_c"))) /
                 F.exp((17.625 * F.col("air_temp_c")) /
                       (243.04 + F.col("air_temp_c"))))
)

# Fill weather nulls with median-ish defaults (no per-airport medians for time)
joined = joined.fillna({
    "air_temp_c": 15.0,
    "dew_point_c": 5.0,
    "slp_hpa": 1013.0,
    "wind_speed_ms": 3.0,
    "sky_cond": 4,
    "precip_1h_mm": 0.0,
    "rel_humidity": 60.0,
})

# Drop rows with null target
joined = joined.filter(F.col("dep_del15").isNotNull())

# ---------- 6. Time-based train/test split ----------
final_cols = [
    "year", "month", "day", "dow", "crs_dep_hour",
    "carrier", "origin", "dest",
    "distance", "crs_elapsed",
    "hour_sin", "hour_cos", "month_sin", "month_cos", "dow_sin", "dow_cos",
    "is_weekend", "is_peak_hour", "season",
    "air_temp_c", "dew_point_c", "slp_hpa", "wind_speed_ms",
    "sky_cond", "precip_1h_mm", "rel_humidity",
    "is_freezing", "is_overcast", "is_strong_wind", "is_precip",
    "weather_severity",
    "dep_delay", "dep_del15"
]

train = joined.filter(F.col("year").isin([2023, 2024])).select(final_cols)
test  = joined.filter(F.col("year") == 2025).select(final_cols)

print(f"Train rows: {train.count():,}")
print(f"Test rows: {test.count():,}")
print(f"Train delay rate: {train.agg(F.mean('dep_del15')).first()[0]:.3f}")
print(f"Test delay rate:  {test.agg(F.mean('dep_del15')).first()[0]:.3f}")

train.write.mode("overwrite").parquet(OUTPUT_TRAIN)
test.write.mode("overwrite").parquet(OUTPUT_TEST)

print("Done.")
spark.stop()