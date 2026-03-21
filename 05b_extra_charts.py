import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

def load_spark_csv(folder_name):
    path = os.path.join("flight_data", folder_name, "*.csv")
    files = glob.glob(path)
    return pd.read_csv(files[0])

print("Loading new data...")
df_airlines = load_spark_csv("eda_airline_delays")
df_heatmap = load_spark_csv("eda_heatmap_data")

sns.set_theme(style="whitegrid", context="talk")

# ==========================================
# CHART 4: Airline Performance
# ==========================================
print("Generating Airline chart...")
plt.figure(figsize=(12, 6))
df_airlines_sorted = df_airlines.sort_values(by="Avg_Delay", ascending=False).head(15)
sns.barplot(data=df_airlines_sorted, x="Avg_Delay", y="Reporting_Airline", hue="Reporting_Airline", palette="flare", legend=False)
plt.title("Carrier Performance: Average Delay by Airline", pad=20, fontweight='bold')
plt.xlabel("Average Delay (Minutes)")
plt.ylabel("Airline Code")
plt.tight_layout()
plt.savefig("flight_data/Chart_4_Airline_Delays.png", dpi=300)
plt.close()

# ==========================================
# CHART 5: The Temporal Heatmap
# ==========================================
print("Generating Heatmap...")
plt.figure(figsize=(14, 6))

# Map DayOfWeek numbers to actual names (BTS standard: 1=Mon, 7=Sun)
day_map = {1:'Mon', 2:'Tue', 3:'Wed', 4:'Thu', 5:'Fri', 6:'Sat', 7:'Sun'}
df_heatmap['Day_Name'] = df_heatmap['DayOfWeek'].map(day_map)

# Create a 2D matrix for the heatmap
pivot_data = df_heatmap.pivot(index='Day_Name', columns='Hour', values='Avg_Delay')
# Force chronological order for the Y-axis
pivot_data = pivot_data.reindex(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])

# Draw the heatmap
sns.heatmap(pivot_data, cmap="YlOrRd", annot=False, linewidths=.5)
plt.title("Temporal Hotspots: Delays by Day of Week and Hour", pad=20, fontweight='bold')
plt.xlabel("Hour of Day (24H Format)")
plt.ylabel("Day of the Week")
plt.tight_layout()
plt.savefig("flight_data/Chart_5_Temporal_Heatmap.png", dpi=300)
plt.close()

print("Boom! Two new advanced charts saved to flight_data.")