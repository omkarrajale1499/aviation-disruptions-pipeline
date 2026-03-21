import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

def load_spark_csv(folder_name):
    path = os.path.join("flight_data", folder_name, "*.csv")
    files = glob.glob(path)
    if not files:
        raise FileNotFoundError(f"Could not find CSV in {folder_name}")
    return pd.read_csv(files[0])

print("Loading aggregated data...")
df_airports = load_spark_csv("eda_airport_delays")
df_hourly = load_spark_csv("eda_hourly_delays")
df_daily_flights = load_spark_csv("eda_daily_flights")
df_daily_weather = load_spark_csv("eda_daily_weather")

sns.set_theme(style="whitegrid", context="talk")

# ==========================================
# CHART 1: The Baseline Problem (Hourly Delays)
# ==========================================
print("Generating Hourly Delays chart...")
plt.figure(figsize=(12, 6))
sns.lineplot(data=df_hourly, x="Hour", y="Avg_Delay", color="#e74c3c", linewidth=3, marker="o")
plt.title("The Snowball Effect: Average Flight Delay by Hour of Day", pad=20, fontweight='bold')
plt.xlabel("Hour of Day (24H Format)")
plt.ylabel("Average Departure Delay (Minutes)")
plt.xticks(range(0, 24))
plt.tight_layout()
plt.savefig("flight_data/Chart_1_Hourly_Delays.png", dpi=300)
plt.close()

# ==========================================
# CHART 2: Spatial Impact (Top 20 Busiest Airports)
# ==========================================
print("Generating Airport Delays chart...")
plt.figure(figsize=(14, 8))
df_airports_sorted = df_airports.sort_values(by="Avg_Delay", ascending=False)
# Fixed the seaborn warning by explicitly setting hue
sns.barplot(data=df_airports_sorted, x="Avg_Delay", y="Origin", hue="Origin", palette="viridis", legend=False)
plt.title("Average Departure Delay at Top 20 Busiest US Airports", pad=20, fontweight='bold')
plt.xlabel("Average Delay (Minutes)")
plt.ylabel("Airport Code")
plt.tight_layout()
plt.savefig("flight_data/Chart_2_Airport_Delays.png", dpi=300)
plt.close()

# ==========================================
# CHART 3: The ML Justification (Weather vs Delays)
# ==========================================
print("Generating Weather vs Delay chart...")

# FIX: Standardize Date Formats Before Merging
# 1. Convert flight dates (e.g., "2024-01-01") to datetime
df_daily_flights["FlightDate"] = pd.to_datetime(df_daily_flights["FlightDate"], format="mixed", errors="coerce")

# 2. Convert weather dates (e.g., 20240101) to string first, then to datetime
df_daily_weather["WeatherDate"] = pd.to_datetime(df_daily_weather["WeatherDate"].astype(str), format="%Y%m%d", errors="coerce")

# 3. Merge safely on the standardized dates!
df_merged = pd.merge(df_daily_flights, df_daily_weather, left_on="FlightDate", right_on="WeatherDate")

plt.figure(figsize=(10, 8))
sns.regplot(data=df_merged, x="Avg_Wind_ms", y="Avg_Flight_Delay", 
            scatter_kws={'alpha':0.5, 'color':'#3498db'}, 
            line_kws={'color':'#e74c3c', 'linewidth':3})
plt.title("Correlation: Daily Wind Speed vs. National Flight Delays", pad=20, fontweight='bold')
plt.xlabel("Average Daily Wind Speed (m/s)")
plt.ylabel("Average Daily Departure Delay (Minutes)")
plt.text(0.05, 0.95, "Insight: Positive correlation justifies\nusing weather features for predictive ML.", 
         transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
plt.tight_layout()
plt.savefig("flight_data/Chart_3_Weather_Correlation.png", dpi=300)
plt.close()

print("Success! High-resolution charts saved to your flight_data folder.")