"""
Export small CSV inputs for the dedicated Python plotting container.

Spark reads the full HDFS datasets and writes compact local CSV files under
/data/plot_inputs, which is bind-mounted to the host and the plotter container.
"""
import csv
import os

from pyspark.ml.functions import vector_to_array
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


HDFS = "hdfs://namenode:9000"
TRAIN = f"{HDFS}/user/team/flight_delay/processed/train"
PRED = f"{HDFS}/user/team/flight_delay/predictions/rf_test"
MODEL = f"{HDFS}/user/team/flight_delay/models/rf_classifier"
OUT = "/data/plot_inputs"


def write_rows(name, columns, rows):
    path = os.path.join(OUT, name)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([row[col] for col in columns])
    print(f"Wrote {path}")


os.makedirs(OUT, exist_ok=True)

spark = (
    SparkSession.builder
    .appName("export_plot_data")
    .config("spark.sql.shuffle.partitions", "64")
    .getOrCreate()
)

train = spark.read.parquet(TRAIN)
pred = spark.read.parquet(PRED).select(
    F.col("dep_del15").cast("double").alias("dep_del15"),
    vector_to_array("probability")[1].alias("p_delay"),
)

# Keep this sample small enough for local plotting.
delay_rows = (
    train.filter(F.abs(F.col("dep_delay")) < 300)
    .select("dep_delay", "season")
    .sample(False, 0.01, seed=42)
    .limit(100000)
    .collect()
)
write_rows("delay_distribution.csv", ["dep_delay", "season"], delay_rows)

airport_rows = (
    train.groupBy("origin", "crs_dep_hour")
    .agg(F.mean("dep_del15").alias("delay_rate"), F.count("*").alias("n"))
    .filter(F.col("n") >= 500)
    .collect()
)
origin_totals = {}
for row in airport_rows:
    origin_totals[row["origin"]] = origin_totals.get(row["origin"], 0) + row["n"]
top20 = {origin for origin, _ in sorted(origin_totals.items(), key=lambda item: -item[1])[:20]}
airport_rows = [row for row in airport_rows if row["origin"] in top20]
write_rows(
    "airport_fingerprint.csv",
    ["origin", "crs_dep_hour", "delay_rate", "n"],
    airport_rows,
)

pred_rows = pred.sample(False, 0.25, seed=42).limit(200000).collect()
write_rows("predictions_sample.csv", ["dep_del15", "p_delay"], pred_rows)

weather_rows = (
    train.filter(F.col("weather_severity").isNotNull())
    .filter(F.col("weather_severity").cast("int").isNotNull())
    .withColumn("weather_severity", F.col("weather_severity").cast("int"))
    .groupBy("weather_severity")
    .agg(F.mean("dep_del15").alias("delay_rate"), F.count("*").alias("n"))
    .orderBy("weather_severity")
    .collect()
)
print("Weather severity data:")
for row in weather_rows:
    print(row)
write_rows("weather_severity.csv", ["weather_severity", "delay_rate", "n"], weather_rows)

calibration_rows = (
    pred.withColumn("bin", F.least(F.floor(F.col("p_delay") * 10), F.lit(9)))
    .groupBy("bin")
    .agg(
        F.mean("p_delay").alias("p_mean"),
        F.mean("dep_del15").alias("y_mean"),
        F.count("*").alias("n"),
    )
    .orderBy("bin")
    .collect()
)
write_rows("calibration.csv", ["bin", "p_mean", "y_mean", "n"], calibration_rows)

# Feature importance from the saved PySpark Random Forest pipeline.
try:
    pipeline = PipelineModel.load(MODEL)
    rf_model = pipeline.stages[-1]
    importances = rf_model.featureImportances.toArray().tolist()

    cat_cols = ["carrier", "origin_b", "dest_b", "season"]
    indexer_stages = pipeline.stages[:len(cat_cols)]
    encoder_stages = pipeline.stages[len(cat_cols):len(cat_cols) * 2]

    feature_names = []
    categories = []
    for cat_col, indexer, encoder in zip(cat_cols, indexer_stages, encoder_stages):
        labels = list(indexer.labels)
        category_size = encoder.categorySizes[0]
        if len(labels) < category_size:
            labels.extend([f"__invalid_{i}" for i in range(category_size - len(labels))])
        ohe_size = category_size - 1 if encoder.getDropLast() else category_size
        for label in labels[:ohe_size]:
            feature_names.append(f"{cat_col}={label}")
            categories.append(
                "carrier" if cat_col == "carrier"
                else "airport" if cat_col in ("origin_b", "dest_b")
                else "temporal/flight"
            )

    numeric_features = [
        "distance", "crs_elapsed",
        "hour_sin", "hour_cos", "month_sin", "month_cos", "dow_sin", "dow_cos",
        "is_weekend", "is_peak_hour",
        "air_temp_c", "dew_point_c", "slp_hpa", "wind_speed_ms",
        "sky_cond", "precip_1h_mm", "rel_humidity",
        "is_freezing", "is_overcast", "is_strong_wind", "is_precip",
        "weather_severity",
    ]
    weather_terms = {
        "air_temp_c", "dew_point_c", "slp_hpa", "wind_speed_ms", "sky_cond",
        "precip_1h_mm", "rel_humidity", "is_freezing", "is_overcast",
        "is_strong_wind", "is_precip", "weather_severity",
    }
    for feature in numeric_features:
        feature_names.append(feature)
        categories.append("weather" if feature in weather_terms else "temporal/flight")

    while len(feature_names) < len(importances):
        feature_names.append(f"feature_{len(feature_names)}")
        categories.append("other")

    feature_rows = [
        {
            "feature": feature_names[i],
            "importance": float(importance),
            "category": categories[i],
        }
        for i, importance in enumerate(importances)
    ]
    feature_rows = sorted(feature_rows, key=lambda row: row["importance"], reverse=True)[:30]
    write_rows("feature_importance.csv", ["feature", "importance", "category"], feature_rows)
except Exception as exc:
    print(f"WARNING: Feature importance export skipped: {exc}")

print(f"Plot input CSVs saved to {OUT}")
spark.stop()
