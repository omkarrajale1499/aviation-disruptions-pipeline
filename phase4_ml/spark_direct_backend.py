"""Optional direct Spark ML backend for the Streamlit demo.

This module lets the demo call the saved PySpark PipelineModel directly when
PySpark and HDFS are reachable from the environment running Streamlit. The app
keeps the existing sklearn proxy model as a fallback for presentation safety.
"""
import math
import os


HDFS_MODEL_PATH = os.environ.get(
    "SPARK_RF_MODEL_PATH",
    "hdfs://namenode:9000/user/team/flight_delay/models/rf_classifier",
)


def _rel_humidity(air_temp_c, dew_point_c):
    if air_temp_c is None or dew_point_c is None:
        return 60.0
    return float(
        100
        * math.exp((17.625 * dew_point_c) / (243.04 + dew_point_c))
        / math.exp((17.625 * air_temp_c) / (243.04 + air_temp_c))
    )


def _season(month):
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "fall"


def _labels_by_input_col(pipeline):
    labels = {}
    for stage in pipeline.stages:
        if hasattr(stage, "getInputCol") and hasattr(stage, "labels"):
            labels[stage.getInputCol()] = set(stage.labels)
    return labels


def _bucket(value, valid_labels):
    if value in valid_labels:
        return value
    if "OTHER" in valid_labels:
        return "OTHER"
    return value


def load_backend():
    from pyspark.ml import PipelineModel
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder
        .appName("streamlit_direct_rf_prediction")
        .master(os.environ.get("SPARK_DEMO_MASTER", "local[*]"))
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    pipeline = PipelineModel.load(HDFS_MODEL_PATH)
    labels = _labels_by_input_col(pipeline)
    return spark, pipeline, labels


def predict_probability(backend, inputs):
    spark, pipeline, labels = backend

    month = int(inputs["month"])
    dow = int(inputs["dow"])
    dep_hour = int(inputs["dep_hour"])
    air_temp_c = float(inputs["air_temp_c"])
    wind_speed_ms = float(inputs["wind_speed_ms"])
    precip_1h_mm = float(inputs["precip_1h_mm"])
    dew_point_c = min(air_temp_c, air_temp_c - 8.0)
    sky_cond = 8 if inputs["weather_severity"] >= 2 else 5 if inputs["weather_severity"] == 1 else 2

    is_freezing = int(air_temp_c < 0)
    is_overcast = int(sky_cond >= 7)
    is_strong_wind = int(wind_speed_ms > 10)
    is_precip = int(precip_1h_mm > 0.1)

    row = {
        "carrier": _bucket(inputs["carrier"], labels.get("carrier", set())),
        "origin_b": _bucket(inputs["origin"], labels.get("origin_b", set())),
        "dest_b": _bucket(inputs["dest"], labels.get("dest_b", set())),
        "season": _season(month),
        "distance": float(inputs["distance"]),
        "crs_elapsed": float(inputs["crs_elapsed"]),
        "hour_sin": math.sin(dep_hour * (2 * math.pi / 24)),
        "hour_cos": math.cos(dep_hour * (2 * math.pi / 24)),
        "month_sin": math.sin(month * (2 * math.pi / 12)),
        "month_cos": math.cos(month * (2 * math.pi / 12)),
        "dow_sin": math.sin(dow * (2 * math.pi / 7)),
        "dow_cos": math.cos(dow * (2 * math.pi / 7)),
        "is_weekend": int(dow >= 6),
        "is_peak_hour": int(dep_hour in [7, 8, 17, 18, 19]),
        "air_temp_c": air_temp_c,
        "dew_point_c": dew_point_c,
        "slp_hpa": 1013.0,
        "wind_speed_ms": wind_speed_ms,
        "sky_cond": int(sky_cond),
        "precip_1h_mm": precip_1h_mm,
        "rel_humidity": _rel_humidity(air_temp_c, dew_point_c),
        "is_freezing": is_freezing,
        "is_overcast": is_overcast,
        "is_strong_wind": is_strong_wind,
        "is_precip": is_precip,
        "weather_severity": int(inputs["weather_severity"]),
    }

    input_df = spark.createDataFrame([row])
    prediction = pipeline.transform(input_df)
    probability = prediction.select("probability").first()[0]
    return float(probability[1])
