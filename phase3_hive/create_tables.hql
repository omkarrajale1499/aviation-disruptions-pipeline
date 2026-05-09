CREATE DATABASE IF NOT EXISTS flight_delay;
USE flight_delay;

DROP TABLE IF EXISTS train_data;
CREATE EXTERNAL TABLE train_data (
    year INT, month INT, day INT, dow INT, crs_dep_hour INT,
    carrier STRING, origin STRING, dest STRING,
    distance DOUBLE, crs_elapsed DOUBLE,
    hour_sin DOUBLE, hour_cos DOUBLE, month_sin DOUBLE, month_cos DOUBLE,
    dow_sin DOUBLE, dow_cos DOUBLE,
    is_weekend INT, is_peak_hour INT, season STRING,
    air_temp_c DOUBLE, dew_point_c DOUBLE, slp_hpa DOUBLE, wind_speed_ms DOUBLE,
    sky_cond INT, precip_1h_mm DOUBLE, rel_humidity DOUBLE,
    is_freezing INT, is_overcast INT, is_strong_wind INT, is_precip INT,
    weather_severity INT,
    dep_delay DOUBLE, dep_del15 DOUBLE
)
STORED AS PARQUET
LOCATION 'hdfs://namenode:9000/user/team/flight_delay/processed/train';

DROP TABLE IF EXISTS test_data;
CREATE EXTERNAL TABLE test_data LIKE train_data;
ALTER TABLE test_data SET LOCATION 'hdfs://namenode:9000/user/team/flight_delay/processed/test';

-- Combined view for cross-year queries
CREATE OR REPLACE VIEW all_flights AS
SELECT * FROM train_data UNION ALL SELECT * FROM test_data;

SHOW TABLES;
SELECT COUNT(*) AS train_rows FROM train_data;
SELECT COUNT(*) AS test_rows FROM test_data;