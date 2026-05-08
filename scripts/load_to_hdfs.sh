#!/bin/bash
set -e

echo "Loading flights..."
hdfs dfs -put -f /data/flights/*.csv /user/team/flight_delay/raw/flights/

echo "Loading weather..."
# NOAA files are organized by year subdirectory
for YEAR in 2023 2024 2025; do
    if [ -d "/data/weather/$YEAR" ]; then
        hdfs dfs -mkdir -p /user/team/flight_delay/raw/weather/$YEAR
        hdfs dfs -put -f /data/weather/$YEAR/* /user/team/flight_delay/raw/weather/$YEAR/
    fi
done

# Load the lookup files
hdfs dfs -put -f /data/weather/isd-history.csv /user/team/flight_delay/raw/
hdfs dfs -put -f /data/airports.csv /user/team/flight_delay/raw/

echo "Verifying sizes..."
hdfs dfs -du -s -h /user/team/flight_delay/raw