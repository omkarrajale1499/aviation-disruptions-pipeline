"""
Random Forest classifier for dep_del15.
Single training pass, no CV (time-constrained).
Class imbalance handled via sample weights.
"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import json

HDFS = "hdfs://namenode:9000"
TRAIN = f"{HDFS}/user/team/flight_delay/processed/train"
TEST = f"{HDFS}/user/team/flight_delay/processed/test"
MODEL_OUT = f"{HDFS}/user/team/flight_delay/models/rf_classifier"
PRED_OUT = f"{HDFS}/user/team/flight_delay/predictions/rf_test"
METRICS_OUT = f"{HDFS}/user/team/flight_delay/metrics/rf_classifier"

spark = (SparkSession.builder
         .appName("train_classifier")
         .config("spark.sql.shuffle.partitions", "64")
         .getOrCreate())

train = spark.read.parquet(TRAIN)
test = spark.read.parquet(TEST)

# Add sample weight for class imbalance
pos_rate = train.agg(F.mean("dep_del15")).first()[0]
neg_weight = pos_rate
pos_weight = 1.0 - pos_rate

train = train.withColumn(
    "class_weight",
    F.when(F.col("dep_del15") == 1, F.lit(pos_weight * 2))
     .otherwise(F.lit(neg_weight * 2))
)

# Categorical: carrier, origin (top 30 only), dest (top 30), season
# Bucket rare airports as "OTHER"
top_origins = [r["origin"] for r in
               train.groupBy("origin").count()
                    .orderBy(F.desc("count")).limit(30).collect()]
top_dests = [r["dest"] for r in
             train.groupBy("dest").count()
                  .orderBy(F.desc("count")).limit(30).collect()]

def bucket_rare(df):
    return (df
        .withColumn("origin_b",
                    F.when(F.col("origin").isin(top_origins), F.col("origin"))
                     .otherwise("OTHER"))
        .withColumn("dest_b",
                    F.when(F.col("dest").isin(top_dests), F.col("dest"))
                     .otherwise("OTHER")))

train = bucket_rare(train)
test = bucket_rare(test)

# Build pipeline
cat_cols = ["carrier", "origin_b", "dest_b", "season"]
indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx",
                          handleInvalid="keep") for c in cat_cols]
encoders = [OneHotEncoder(inputCol=f"{c}_idx",
                          outputCol=f"{c}_vec") for c in cat_cols]

num_cols = [
    "distance", "crs_elapsed",
    "hour_sin", "hour_cos", "month_sin", "month_cos", "dow_sin", "dow_cos",
    "is_weekend", "is_peak_hour",
    "air_temp_c", "dew_point_c", "slp_hpa", "wind_speed_ms",
    "sky_cond", "precip_1h_mm", "rel_humidity",
    "is_freezing", "is_overcast", "is_strong_wind", "is_precip",
    "weather_severity",
]

assembler = VectorAssembler(
    inputCols=[f"{c}_vec" for c in cat_cols] + num_cols,
    outputCol="features"
)

rf = RandomForestClassifier(
    labelCol="dep_del15",
    featuresCol="features",
    weightCol="class_weight",
    numTrees=20,
    maxDepth=6,
    maxBins=32,
    seed=42
)

pipeline = Pipeline(stages=indexers + encoders + [assembler, rf])

print("Training...")
model = pipeline.fit(train)

print("Predicting on test...")
preds = model.transform(test)

# Metrics
auc_eval = BinaryClassificationEvaluator(
    labelCol="dep_del15", rawPredictionCol="rawPrediction",
    metricName="areaUnderROC")
pr_eval = BinaryClassificationEvaluator(
    labelCol="dep_del15", rawPredictionCol="rawPrediction",
    metricName="areaUnderPR")
f1_eval = MulticlassClassificationEvaluator(
    labelCol="dep_del15", predictionCol="prediction", metricName="f1")

auc = auc_eval.evaluate(preds)
pr_auc = pr_eval.evaluate(preds)
f1 = f1_eval.evaluate(preds)

# Confusion matrix
cm = (preds.groupBy("dep_del15", "prediction").count()
      .orderBy("dep_del15", "prediction").collect())

print("=" * 50)
print(f"ROC-AUC : {auc:.4f}")
print(f"PR-AUC  : {pr_auc:.4f}")
print(f"F1      : {f1:.4f}")
print("Confusion matrix (actual, predicted, count):")
for r in cm:
    print(f"  {int(r['dep_del15'])} -> {int(r['prediction'])}: {r['count']:,}")
print("=" * 50)

# Feature importances
rf_model = model.stages[-1]
importances = rf_model.featureImportances.toArray()
feature_names = (
    [f"{c}_vec" for c in cat_cols] + num_cols
)
# Note: OHE creates expanded indices; for simplicity we report top numeric features
num_start = sum(1 for _ in cat_cols)  # rough — see report caveat
top_n = sorted(enumerate(importances), key=lambda x: -x[1])[:20]
print("Top 20 feature indices and importances:")
for i, imp in top_n:
    print(f"  idx={i}: {imp:.4f}")

# Save predictions for plotting
(preds.select("dep_del15", "prediction", "probability", "rawPrediction")
      .write.mode("overwrite").parquet(PRED_OUT))

# Save metrics to a small JSON for the report
metrics = {"roc_auc": auc, "pr_auc": pr_auc, "f1": f1}
print("METRICS_JSON:", json.dumps(metrics))
spark.createDataFrame([metrics]).write.mode("overwrite").json(METRICS_OUT)
print(f"Metrics saved to {METRICS_OUT}")

model.write().overwrite().save(MODEL_OUT)
print(f"Model saved to {MODEL_OUT}")
spark.stop()