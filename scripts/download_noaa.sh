#!/bin/bash
# Downloads NOAA ISD-Lite for US airport stations, 2023-2025
# Filters to stations with ICAO starting with K (continental US airports)

set -e

OUTPUT_DIR="/data/weather"
YEARS=(2023 2024 2025)
HISTORY_FILE="$OUTPUT_DIR/isd-history.csv"

mkdir -p "$OUTPUT_DIR"

# Step 1: Get the station history file
echo "Downloading station history..."
curl -s -o "$HISTORY_FILE" https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv

# Step 2: Extract US airport stations (ICAO starts with K, country = US)
# Format: USAF,WBAN,STATION NAME,CTRY,ST,ICAO,LAT,LON,ELEV,BEGIN,END
echo "Filtering US airport stations..."
US_STATIONS=$(awk -F',' 'NR>1 && $4=="\"US\"" && $6 ~ /^"K/ {gsub(/"/,"",$1); gsub(/"/,"",$2); print $1"-"$2}' "$HISTORY_FILE")

STATION_COUNT=$(echo "$US_STATIONS" | wc -l)
echo "Found $STATION_COUNT US airport stations"

# Step 3: Download each station-year
for YEAR in "${YEARS[@]}"; do
    YEAR_DIR="$OUTPUT_DIR/$YEAR"
    mkdir -p "$YEAR_DIR"
    echo "Downloading year $YEAR..."
    
    COUNT=0
    for STATION in $US_STATIONS; do
        URL="https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/$YEAR/${STATION}-${YEAR}.gz"
        OUTFILE="$YEAR_DIR/${STATION}-${YEAR}.gz"
        
        # Skip if already downloaded
        if [ -f "$OUTFILE" ]; then
            continue
        fi
        
        # Download silently, allow failures (some stations don't have data every year)
        curl -s -f -o "$OUTFILE" "$URL" || true
        
        COUNT=$((COUNT+1))
        if [ $((COUNT % 100)) -eq 0 ]; then
            echo "  Progress: $COUNT stations"
        fi
    done
    
    # Decompress all .gz files
    echo "Decompressing $YEAR..."
    gunzip -f "$YEAR_DIR"/*.gz 2>/dev/null || true
done

echo "Done. Files in $OUTPUT_DIR:"
for Y in "${YEARS[@]}"; do
    ls "$OUTPUT_DIR/$Y" 2>/dev/null | wc -l | xargs echo "  $Y files:"
done