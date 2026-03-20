# ✈️ Aviation Disruptions: A Big Data Pipeline

> **Predicting flight delays using large-scale climate data and Apache Spark.** > Developed for the Master's in Applied Data Intelligence program at San Jose State University (SJSU).

## 📋 Project Overview
Flight delays cause massive logistical bottlenecks across the national airspace system. While many delays are attributed to operational constraints, localized extreme weather is a severe, highly variable disruptor. 

This project establishes a distributed Big Data pipeline to ingest, clean, and analyze over 10GB of modern aviation and climate data. By leveraging Hadoop and PySpark, we built a robust foundation to explore the spatial-temporal correlation between weather events and flight departure delays, setting the stage for predictive Machine Learning modeling.

## 🗄️ Dataset Architecture
We synthesized two massive, disparate datasets representing the post-COVID modern aviation baseline (2024–2025):
* **Bureau of Transportation Statistics (BTS):** 24 months of domestic flight performance data (~15 million rows, CSV format).
* **NOAA Integrated Surface Database (ISD):** Hourly global weather observations from thousands of stations (Fixed-width text format, compressed `.gz`).

## 🛠️ Technology Stack
* **Storage:** Hadoop Distributed File System (HDFS)
* **Compute:** Apache Spark (PySpark)
* **Infrastructure:** Dockerized multi-node cluster (NameNode, DataNodes, Spark Master/Workers)
* **Analytics & Visualization:** Pandas, Matplotlib, Seaborn
* **Data Lake Format:** Partitioned Apache Parquet

## ⚙️ System Pipeline
1.  **Ingestion:** Raw CSVs and fixed-width text files are loaded into the HDFS Raw Zone.
2.  **Distributed Processing:** PySpark extracts weather features (slicing fixed-width strings, handling `9999` nulls, scaling temperatures) and cleans flight data (filtering cancelled/diverted flights).
3.  **Data Lake:** Cleaned data is written back to HDFS as compressed, partitioned Parquet blocks to overcome the Hadoop "Small File Problem".
4.  **Aggregation:** PySpark reduces millions of rows into lightweight summary metrics.
5.  **EDA:** Local Python scripts generate presentation-ready visualizations.

## 📊 Exploratory Data Analysis (EDA) Insights
Our preliminary EDA revealed critical complexities in the dataset:
* **The Snowball Effect:** Morning flights operate efficiently, but delays compound exponentially throughout the day, peaking at 8:00 PM.
* **Spatial Bottlenecks:** Delays are heavily concentrated at specific hubs (e.g., DFW, CLT) rather than distributed evenly.
* **The "Washout" Phenomenon:** When aggregating weather and delays on a *national daily scale*, correlations appear flat. This proves that localized weather events (e.g., a blizzard in Chicago) are "washed out" by calm weather elsewhere.

## 🚀 Future Scope (Final Project)
The insights from our EDA prove that simple BI aggregations are insufficient. For the final phase of this project, we will build a distributed Machine Learning pipeline using **Spark MLlib**. We will execute complex spatial-temporal joins (matching specific airports to specific weather stations by the exact hour) to train a non-linear predictive model (e.g., Random Forest) capable of anticipating localized weather-driven delays.

## 👥 Team Contributors
* **Omkar [Last Name]** - Data Engineering & Infrastructure
* **[Teammate 2 Name]** - PySpark Development
* **[Teammate 3 Name]** - Data Analytics & Visualization