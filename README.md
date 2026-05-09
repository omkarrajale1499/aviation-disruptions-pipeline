# Flight Delay Prediction with Spark, Hive, HDFS, and Streamlit

This project implements an end-to-end big-data pipeline for predicting U.S. flight departure delays. It combines BTS flight data with NOAA ISD-Lite weather observations, processes the data with PySpark on HDFS, performs EDA with Hive, trains a Spark ML Random Forest classifier, generates presentation plots, and provides a Streamlit demo app.

## Project Overview

The pipeline predicts whether a flight will be delayed by at least 15 minutes (`dep_del15`) using flight schedule, airport, carrier, calendar, and weather features.

High-level flow:

```text
BTS flight CSVs + NOAA weather files
        -> HDFS raw storage
        -> PySpark ingestion
        -> Flight/weather join + feature engineering
        -> Hive EDA
        -> Spark ML Random Forest training
        -> Plot generation
        -> Streamlit demo app
```

## Tech Stack

- Docker / Docker Compose
- Hadoop HDFS
- Apache Spark / PySpark
- Apache Hive + PostgreSQL metastore
- Spark MLlib
- Python, pandas, scikit-learn
- Plotly, Matplotlib, Seaborn
- Streamlit

## Repository Structure

```text
.
├── docker-compose.yml
├── Dockerfile.spark-master
├── scripts/
│   ├── download_noaa.sh
│   └── load_to_hdfs.sh
├── phase2_pyspark/
│   ├── ingest_flights.py
│   ├── ingest_weather.py
│   └── join_and_features.py
├── phase3_hive/
│   ├── create_tables.hql
│   ├── eda_queries.hql
│   └── eda_results.txt
├── phase4_ml/
│   ├── train_classifier.py
│   ├── export_model_for_demo.py
│   ├── train_demo_model_local.py
│   ├── spark_direct_backend.py
│   ├── app.py
│   └── requirements_demo.txt
├── phase5_viz/
│   ├── export_plot_data.py
│   ├── make_plots_local.py
│   ├── make_plots.py
│   └── requirements.txt
├── app.py
└── README.md

```

## Docker Services

The project runs a local multi-container cluster:

- `namenode`: HDFS NameNode
- `datanode-1`, `datanode-2`: HDFS DataNodes
- `spark-master`: Spark master
- `spark-worker-1`, `spark-worker-2`: Spark workers
- `hive-metastore-postgresql`: PostgreSQL backend for Hive metastore
- `hive-metastore`: Hive metastore service
- `hive-server`: HiveServer2
- `plotter`: Python plotting container

Useful URLs:

- HDFS NameNode UI: `http://localhost:9870`
- Spark Master UI: `http://localhost:8080`
- Spark App UI: `http://localhost:4040`
- HiveServer2 UI: `http://localhost:10002`
- Streamlit App: `http://localhost:8501`

## Setup

Build the custom Spark image:

```powershell
docker-compose build spark-master
```

Start the cluster:

```powershell
docker-compose up -d
```

Check services:

```powershell
docker-compose ps
```

## Data Layout

Expected local input folders:

```text
data/flights/
data/weather/
data/airports.csv
```

Main HDFS paths:

```text
/user/team/flight_delay/raw
/user/team/flight_delay/staging/flights
/user/team/flight_delay/staging/weather
/user/team/flight_delay/processed/train
/user/team/flight_delay/processed/test
/user/team/flight_delay/models/rf_classifier
/user/team/flight_delay/predictions/rf_test
/user/team/flight_delay/metrics/rf_classifier
```

## Running the Pipeline

### 1. Download NOAA Weather Data

```powershell
docker exec namenode bash /scripts/download_noaa.sh
```

This downloads NOAA ISD-Lite weather files for 2023, 2024, and 2025.

### 2. Load Raw Data to HDFS

```powershell
docker exec namenode bash /scripts/load_to_hdfs.sh
```

Verify raw weather years:

```powershell
docker exec namenode /opt/hadoop-3.2.1/bin/hdfs dfs -ls /user/team/flight_delay/raw/weather
```

Expected:

```text
2023
2024
2025
```

### 3. Ingest Flights

```powershell
docker exec spark-master /spark/bin/spark-submit /opt/project/phase2_pyspark/ingest_flights.py
```

This reads raw BTS flight CSV files, standardizes columns, removes cancelled/diverted flights, removes rows with missing departure delay, builds scheduled departure timestamps, and writes staging Parquet.

### 4. Ingest Weather

```powershell
docker exec spark-master /spark/bin/spark-submit /opt/project/phase2_pyspark/ingest_weather.py
```

This reads NOAA ISD-Lite files, extracts station IDs, replaces `-9999` missing values with nulls, scales raw weather values, creates UTC timestamps, and writes staging Parquet.

### 5. Join Flights and Weather + Engineer Features

```powershell
docker exec spark-master /spark/bin/spark-submit --master spark://spark-master:7077 --executor-cores 1 --executor-memory 2g --driver-memory 2g --conf spark.sql.shuffle.partitions=8 --conf spark.network.timeout=600s --conf spark.executor.heartbeatInterval=60s --conf spark.sql.adaptive.enabled=false --conf spark.speculation=false --conf spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version=1 /opt/project/phase2_pyspark/join_and_features.py
```

The join is done by:

- Mapping each flight origin airport to the nearest NOAA station using latitude/longitude.
- Joining on `origin_station = station_id`.
- Joining on matching scheduled departure hour and weather observation hour.

Recent verified output:

```text
Train rows: 13,288,973
Test rows: 6,674,262
Train delay rate: 0.206
Test delay rate: 0.219
```

## Hive EDA

Create Hive tables:

```powershell
docker exec hive-server beeline -u jdbc:hive2://localhost:10000 -f /opt/project/phase3_hive/create_tables.hql
```

Run EDA queries:

```powershell
docker exec hive-server beeline -u jdbc:hive2://localhost:10000 -f /opt/project/phase3_hive/eda_queries.hql
```

Save results locally:

```powershell
docker exec hive-server beeline -u jdbc:hive2://localhost:10000 -f /opt/project/phase3_hive/eda_queries.hql 2>&1 | Tee-Object -FilePath phase3_hive\eda_results.txt
```

EDA queries include:

- Annual delay statistics
- Worst airports by delay rate
- Delay rate by season and hour
- Carrier comparison
- Weather severity impact
- Worst routes
- Monthly trend

## Model Training

Train the Spark ML Random Forest classifier:

```powershell
docker exec spark-master /spark/bin/spark-submit --master spark://spark-master:7077 --executor-cores 1 --executor-memory 2g --driver-memory 2g --conf spark.sql.shuffle.partitions=8 --conf spark.network.timeout=600s --conf spark.executor.heartbeatInterval=60s --conf spark.sql.adaptive.enabled=false /opt/project/phase4_ml/train_classifier.py
```

Model details:

- Algorithm: Random Forest classifier
- Framework: PySpark MLlib
- Target: `dep_del15`
- Train years: 2023 and 2024
- Test year: 2025
- Categorical features: carrier, origin bucket, destination bucket, season
- Numeric features: distance, elapsed time, cyclical time features, weather values, weather severity

Outputs:

```text
/user/team/flight_delay/models/rf_classifier
/user/team/flight_delay/predictions/rf_test
/user/team/flight_delay/metrics/rf_classifier
```

View metrics:

```powershell
docker exec namenode /opt/hadoop-3.2.1/bin/hdfs dfs -cat /user/team/flight_delay/metrics/rf_classifier/*.json
```

## Visualizations

Export plot input CSVs:

```powershell
docker exec spark-master /spark/bin/spark-submit /opt/project/phase5_viz/export_plot_data.py
```

Generate plots:

```powershell
docker-compose run --rm plotter
```

Outputs are written to:

```text
data/plots/
```

Main plots:

- `01_delay_distribution_v2.png`
- `02_airport_fingerprint.png`
- `03a_roc.png`
- `03b_pr.png`
- `04_weather_severity.png`
- `05_calibration_v2.png`
- `06_feature_importance.png`

## Streamlit Demo

Export demo training data from Spark:

```powershell
docker exec spark-master /spark/bin/spark-submit /opt/project/phase4_ml/export_model_for_demo.py
```

Train local demo model:

```powershell
cd phase4_ml
pip install -r requirements_demo.txt
python train_demo_model_local.py
```

Run the app:

```powershell
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

The Streamlit app provides:

- Live delay probability prediction
- Risk level card
- Airport historical comparison
- Key driver callouts
- Weather severity what-if chart
- EDA insights
- Model performance summary

## Data Cleaning Summary

Flight cleaning:

- Standardize raw BTS column names.
- Cast numeric fields.
- Remove cancelled flights.
- Remove diverted flights.
- Remove rows with missing `dep_delay`.
- Build scheduled departure timestamp and hour fields.

Weather cleaning:

- Parse NOAA ISD-Lite whitespace-separated files.
- Extract `station_id` from filename.
- Replace `-9999` missing values with null.
- Scale raw weather values by dividing by 10.
- Build hourly UTC observation timestamp.

Additional feature-stage handling:

- Drop flights without matched origin weather station.
- Fill missing weather values with reasonable defaults.
- Drop rows with null target `dep_del15`.
- Create weather flags and `weather_severity`.

## Safe Shutdown

To stop containers without losing HDFS data:

```powershell
docker-compose down
```

Do not use this unless you intentionally want to delete Docker volumes:

```powershell
docker-compose down -v
```

Restart:

```powershell
docker-compose up -d
```

## Troubleshooting

If `spark-submit` is not found:

```powershell
docker exec spark-master /spark/bin/spark-submit ...
```

If weather staging misses 2025:

- Confirm `scripts/load_to_hdfs.sh` includes `2025`.
- Reload weather to HDFS.
- Rerun `ingest_weather.py`.
- Rerun `join_and_features.py`.

If Spark write fails with `_temporary` HDFS errors:

```powershell
docker exec namenode /opt/hadoop-3.2.1/bin/hdfs dfs -rm -r -skipTrash /user/team/flight_delay/staging/weather
docker exec spark-master /spark/bin/spark-submit /opt/project/phase2_pyspark/ingest_weather.py
```

If Streamlit dependency errors occur:

```powershell
cd phase4_ml
pip install -r requirements_demo.txt
```

## Notes for Git

Do not commit generated artifacts unless required:

```text
data/
phase4_ml/model_demo.pkl
phase4_ml/demo_metadata.json
phase4_ml/demo_training_sample.csv
phase4_ml/demo_metadata_base.json
__pycache__/
*.pyc
```

These files can be recreated by running the pipeline.

