"""
app.py — Flight Delay Prediction Demo
======================================
Run: streamlit run app.py

Requires (same directory):
  model_demo.pkl
  demo_metadata.json

No Spark, no Docker, no HDFS needed at demo time.
"""
import streamlit as st
import pickle, json, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import spark_direct_backend
except Exception:
    spark_direct_backend = None

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Flight Delay Predictor",
    page_icon="airplane",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@500;600&display=swap');

:root {
    --bg: #f4f7fb;
    --panel: #ffffff;
    --ink: #102033;
    --muted: #667085;
    --line: #d9e2ef;
    --blue: #2563eb;
    --green: #16803c;
    --amber: #b7791f;
    --red: #c53030;
}

.stApp {
    background: linear-gradient(180deg, #eef5fb 0%, var(--bg) 34%, #f8fafc 100%);
    color: var(--ink);
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

h1, h2, h3 {
    color: var(--ink);
    font-weight: 800;
}

section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid var(--line);
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] {
    color: var(--ink) !important;
}
section[data-testid="stSidebar"] small,
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: var(--muted) !important;
}
section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    border: 1px solid var(--line) !important;
    color: var(--ink) !important;
}
section[data-testid="stSidebar"] div[data-baseweb="select"] span {
    color: var(--ink) !important;
}
section[data-testid="stSidebar"] div[data-baseweb="select"] svg {
    fill: var(--ink) !important;
}
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stSelectbox label {
    color: var(--ink) !important;
    font-weight: 700 !important;
}
section[data-testid="stSidebar"] [data-testid="stSliderTickBarMin"],
section[data-testid="stSidebar"] [data-testid="stSliderTickBarMax"] {
    color: var(--muted) !important;
}

.block-container {
    padding-top: 2rem;
    max-width: 1180px;
}

.banner {
    background:
        radial-gradient(circle at 12% 20%, rgba(255,255,255,0.18), transparent 22%),
        linear-gradient(135deg, #0f2b46, #174665 50%, #256d85);
    color: white;
    padding: 30px 34px;
    border-radius: 18px;
    margin-bottom: 22px;
    box-shadow: 0 18px 44px rgba(15, 43, 70, 0.24);
}
.banner h1 { margin: 0; font-size: 2rem; color: #fff; letter-spacing: -0.03em; }
.banner p  { margin: 10px 0 0; color: #d9ecf7; font-size: 0.98rem; }

.pipeline-wrap {
    background: rgba(255, 255, 255, 0.72);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 14px 16px;
    margin-bottom: 18px;
}
.pipeline-label {
    color: var(--muted);
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.pipeline-step {
    display: inline-block;
    background: #eef5ff;
    border: 1px solid #cfe0ff;
    border-radius: 999px;
    padding: 7px 13px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.76rem;
    margin: 3px 4px 3px 0;
    color: #163b73;
}
.arrow {
    color: #7b8794;
    margin: 0 5px 0 1px;
}

.metric-card {
    min-height: 118px;
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 22px 18px;
    text-align: center;
    box-shadow: 0 12px 30px rgba(16, 32, 51, 0.08);
}
.metric-card .value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.18rem;
    font-weight: 700;
    line-height: 1.1;
}
.metric-card .label {
    font-size: 0.72rem;
    color: var(--muted);
    margin-top: 9px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 700;
}
.risk-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    border-radius: 999px;
    padding: 8px 15px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.15rem;
    font-weight: 700;
}
.risk-dot {
    display: inline-block;
    width: 12px;
    height: 12px;
    border-radius: 999px;
}
.risk-low    { color: var(--green); }
.risk-medium { color: var(--amber); }
.risk-high   { color: var(--red); }
.risk-pill.risk-low { background: #e8f7ee; }
.risk-pill.risk-medium { background: #fff4db; }
.risk-pill.risk-high { background: #fdecec; }
.risk-dot.risk-low { background: var(--green); }
.risk-dot.risk-medium { background: var(--amber); }
.risk-dot.risk-high { background: var(--red); }

.driver-card {
    background: #ffffff;
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 18px 20px;
    box-shadow: 0 12px 30px rgba(16, 32, 51, 0.06);
}
.driver-item {
    display: flex;
    gap: 10px;
    align-items: flex-start;
    padding: 7px 0;
    color: var(--ink);
    font-weight: 500;
}
.driver-bullet {
    color: var(--blue);
    font-weight: 800;
}

div[data-testid="stTabs"] button p {
    font-weight: 700;
    color: var(--ink) !important;
}
div[data-testid="stTabs"] button {
    color: var(--ink) !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] p {
    color: #ff4b4b !important;
}
div[data-testid="stTabs"] button[aria-selected="false"] p {
    color: var(--ink) !important;
    opacity: 1 !important;
}
div[data-testid="stTabs"] button:hover p {
    color: #ff4b4b !important;
}
</style>
""", unsafe_allow_html=True)

# ── Load model & metadata ─────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    base = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base, "model_demo.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(base, "demo_metadata.json")) as f:
        meta  = json.load(f)
    return model, meta


@st.cache_resource
def load_spark_backend():
    if spark_direct_backend is None:
        raise RuntimeError("spark_direct_backend.py is unavailable.")
    return spark_direct_backend.load_backend()


def proxy_predict_probability(model, meta, features):
    carrier_enc = meta["carrier_delay_rates"].get(features["carrier"], 0.200)
    origin_enc = meta["airport_delay_rates"].get(features["origin"], 0.215)
    is_weekend = int(features["dow"] >= 6)
    is_peak = int(features["dep_hour"] in [7, 8, 17, 18, 19])
    is_freezing = int(features["air_temp"] < 0)

    x = np.array([[
        features["dep_hour"], features["month"], features["dow"],
        features["distance"], features["crs_elapsed"],
        features["weather_severity"], is_weekend, is_peak,
        features["air_temp"], features["wind_ms"], features["precip"], is_freezing,
        carrier_enc, origin_enc,
    ]])
    return float(model.predict_proba(x)[0][1])


def predict_probability(model, meta, features, backend_mode):
    if backend_mode != "Proxy model":
        try:
            spark_backend = load_spark_backend()
            return (
                spark_direct_backend.predict_probability(
                    spark_backend,
                    {
                        "carrier": features["carrier"],
                        "origin": features["origin"],
                        "dest": features["dest"],
                        "month": features["month"],
                        "dow": features["dow"],
                        "dep_hour": features["dep_hour"],
                        "distance": features["distance"],
                        "crs_elapsed": features["crs_elapsed"],
                        "weather_severity": features["weather_severity"],
                        "air_temp_c": features["air_temp"],
                        "wind_speed_ms": features["wind_ms"],
                        "precip_1h_mm": features["precip"],
                    },
                ),
                "Spark RF pipeline",
                None,
            )
        except Exception as exc:
            if backend_mode == "Spark RF pipeline":
                return proxy_predict_probability(model, meta, features), "Proxy fallback", str(exc)
            return proxy_predict_probability(model, meta, features), "Proxy fallback", str(exc)

    return proxy_predict_probability(model, meta, features), "Proxy model", None

try:
    model, meta = load_model()
    MODEL_OK = True
except FileNotFoundError:
    MODEL_OK = False

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="banner">
  <h1>Airport Disruption Fingerprints</h1>
  <p>Real-time departure delay prediction &nbsp;·&nbsp; 20M+ flights &nbsp;·&nbsp; HDFS + PySpark + Hive + Random Forest</p>
</div>
""", unsafe_allow_html=True)

# ── Pipeline explanation strip ────────────────────────────────────────────────
st.markdown("""
<div class="pipeline-wrap">
  <div class="pipeline-label">Pipeline</div>
  <span class="pipeline-step">1. BTS + NOAA to HDFS</span><span class="arrow">→</span>
  <span class="pipeline-step">2. PySpark ingest + join</span><span class="arrow">→</span>
  <span class="pipeline-step">3. Hive EDA</span><span class="arrow">→</span>
  <span class="pipeline-step">4. Random Forest</span><span class="arrow">→</span>
  <span class="pipeline-step">5. Streamlit demo</span>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Layout: sidebar (inputs) + main (outputs) ─────────────────────────────────
with st.sidebar:
    st.markdown("### Flight Details")
    st.caption("Adjust parameters to predict delay probability 24h in advance.")
    backend_mode = "Proxy model"

    AIRPORT_LIST = sorted([a for a in meta["airports"] if a != "OTHER"])
    CARRIER_LIST = sorted([c for c in meta["carriers"] if c != "OTHER"])

    carrier_labels = {
        "F9": "Frontier (F9)", "B6": "JetBlue (B6)", "NK": "Spirit (NK)",
        "WN": "Southwest (WN)", "AA": "American (AA)", "G4": "Allegiant (G4)",
        "OH": "PSA Airlines (OH)", "AS": "Alaska (AS)", "UA": "United (UA)",
        "HA": "Hawaiian (HA)", "DL": "Delta (DL)", "MQ": "Envoy (MQ)",
        "OO": "SkyWest (OO)", "9E": "Endeavor (9E)", "YX": "Republic (YX)",
    }

    origin = st.selectbox(
        "Origin airport", AIRPORT_LIST,
        index=AIRPORT_LIST.index("ORD") if "ORD" in AIRPORT_LIST else 0,
        help="Departure airport (IATA code)"
    )

    carrier_display = [f"{carrier_labels.get(c, c)}" for c in CARRIER_LIST]
    carrier_idx = st.selectbox(
        "Airline", range(len(CARRIER_LIST)),
        format_func=lambda i: carrier_display[i],
        index=CARRIER_LIST.index("AA") if "AA" in CARRIER_LIST else 0,
    )
    carrier = CARRIER_LIST[carrier_idx]

    dep_hour = st.slider("Scheduled departure hour", 5, 23, 17,
                         help="Local time (0–23)")

    col_m, col_d = st.columns(2)
    with col_m:
        month = st.selectbox("Month", list(range(1, 13)),
                             format_func=lambda m: ["Jan","Feb","Mar","Apr","May","Jun",
                                                     "Jul","Aug","Sep","Oct","Nov","Dec"][m-1],
                             index=6)
    with col_d:
        dow = st.selectbox("Day of week", list(range(1, 8)),
                           format_func=lambda d: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][d-1],
                           index=4)

    st.markdown("---")
    st.markdown("### Weather at Departure")

    weather_severity = st.select_slider(
        "Weather severity", options=[0, 1, 2],
        value=0,
        format_func=lambda v: ["0 — Clear", "1 — Mild disruption", "2 — Multi-condition"][v],
    )

    air_temp = st.slider("Air temperature (°C)", -20, 40, 18)
    wind_ms  = st.slider("Wind speed (m/s)", 0, 25, 5)
    precip   = st.slider("Precipitation (mm/h)", 0.0, 30.0, 0.0, step=0.5)

    st.markdown("---")
    st.markdown("### Route")
    dest = st.selectbox(
        "Destination airport",
        AIRPORT_LIST,
        index=AIRPORT_LIST.index("LAX") if "LAX" in AIRPORT_LIST else 0,
        help="Used by the direct Spark RF backend. The proxy model ignores destination.",
    )
    distance    = st.slider("Distance (miles)", 100, 2800, 800)
    crs_elapsed = st.slider("Scheduled flight time (min)", 45, 360, 150)

    predict_btn = st.button("Predict delay probability", type="primary", width="stretch")

# ── Main panel ────────────────────────────────────────────────────────────────
tab_pred, tab_eda, tab_model = st.tabs(
    ["Prediction", "EDA Insights", "Model Performance"]
)

# ─── TAB 1: Prediction ────────────────────────────────────────────────────────
with tab_pred:
    if not MODEL_OK:
        st.error("❌ `model_demo.pkl` not found. Run `export_model_for_demo.py` on the cluster first.")
        st.stop()

    if predict_btn or True:   # show on load with defaults
        season_map = {12:"winter",1:"winter",2:"winter",
                      3:"spring",4:"spring",5:"spring",
                      6:"summer",7:"summer",8:"summer",
                      9:"fall",10:"fall",11:"fall"}
        season = season_map[month]

        carrier_enc = meta["carrier_delay_rates"].get(carrier, 0.200)
        origin_enc  = meta["airport_delay_rates"].get(origin,  0.215)
        is_weekend  = int(dow >= 6)
        is_peak     = int(dep_hour in [7, 8, 17, 18, 19])
        is_freezing = int(air_temp < 0)
        features = {
            "carrier": carrier,
            "origin": origin,
            "dest": dest,
            "dep_hour": dep_hour,
            "month": month,
            "dow": dow,
            "weather_severity": weather_severity,
            "air_temp": air_temp,
            "wind_ms": wind_ms,
            "precip": precip,
            "distance": distance,
            "crs_elapsed": crs_elapsed,
        }
        prob, backend_used, backend_error = predict_probability(model, meta, features, backend_mode)
        hist_rate = meta["airport_delay_rates"].get(origin, 0.215)

        # ── Metric cards ──────────────────────────────────────────────────────
        col1, col2, col3, col4 = st.columns(4)

        risk_cls = "risk-low" if prob < 0.20 else ("risk-medium" if prob < 0.35 else "risk-high")
        risk_lbl = "LOW" if prob < 0.20 else ("MEDIUM" if prob < 0.35 else "HIGH")

        with col1:
            st.markdown(f"""
            <div class="metric-card">
              <div class="value {risk_cls}">{prob*100:.1f}%</div>
              <div class="label">Delay probability</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
              <div class="risk-pill {risk_cls}">
                <span class="risk-dot {risk_cls}"></span>{risk_lbl}
              </div>
              <div class="label">Risk level</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            delta = (prob - hist_rate) * 100
            sign  = "+" if delta >= 0 else ""
            color = "#cf222e" if delta > 3 else ("#1a7f37" if delta < -3 else "#9a6700")
            st.markdown(f"""
            <div class="metric-card">
              <div class="value" style="color:{color}">{sign}{delta:.1f}pp</div>
              <div class="label">vs {origin} avg ({hist_rate*100:.1f}%)</div>
            </div>""", unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="metric-card">
              <div class="value">{season.capitalize()}</div>
              <div class="label">{["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][dow-1]} {dep_hour:02d}:00</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Probability gauge bar ──────────────────────────────────────────────
        col_g, col_i = st.columns([2, 1])
        with col_g:
            st.markdown("**Delay probability gauge**")
            fig, ax = plt.subplots(figsize=(8.5, 1.55))
            fig.patch.set_facecolor("#ffffff")
            ax.set_facecolor("#ffffff")

            # Background zones
            ax.barh(0, 0.20, left=0.00, height=0.52, color="#d9f2e3", alpha=1.0)
            ax.barh(0, 0.15, left=0.20, height=0.52, color="#fff0c2", alpha=1.0)
            ax.barh(0, 0.65, left=0.35, height=0.52, color="#ffd8d8", alpha=1.0)

            # Indicator
            ax.barh(0, prob, left=0, height=0.6,
                    color=("#16803c" if prob < 0.20 else "#b7791f" if prob < 0.35 else "#c53030"),
                    alpha=0.9)
            ax.axvline(prob, color="#102033", linewidth=2.6)
            label_x = min(prob + 0.015, 0.88)
            ax.text(label_x, 0.08, f"{prob*100:.1f}%", fontsize=12,
                    fontweight="bold", va="center", color="#102033")

            for x_tick, label in [(0.10,"Low"),(0.275,"Medium"),(0.67,"High")]:
                ax.text(x_tick, -0.47, label, ha="center", fontsize=9,
                        color="#475467", fontweight="bold")

            ax.set_xlim(0, 1); ax.set_ylim(-0.6, 0.6)
            ax.axis("off")
            plt.tight_layout(pad=0)
            st.pyplot(fig, width="stretch")
            plt.close()

        with col_i:
            drivers = []
            if dep_hour >= 15:
                drivers.append(f"Late departure hour ({dep_hour}:00)")
            if weather_severity >= 2:
                drivers.append(f"Weather severity = {weather_severity}")
            if is_freezing:
                drivers.append(f"Freezing temperature ({air_temp}°C)")
            if wind_ms > 10:
                drivers.append(f"Strong wind ({wind_ms} m/s)")
            if precip > 0:
                drivers.append(f"Precipitation ({precip} mm/h)")
            if carrier_enc > 0.24:
                drivers.append(f"High-delay carrier ({carrier})")
            if origin_enc > 0.25:
                drivers.append(f"High-delay airport ({origin})")
            if month in [6, 7]:
                drivers.append("Summer peak delay season")
            if not drivers:
                drivers.append("Low-risk operating conditions")
            driver_html = "".join(
                f'<div class="driver-item"><span class="driver-bullet">•</span><span>{d}</span></div>'
                for d in drivers[:5]
            )
            st.markdown(
                f"""
                <div class="driver-card">
                  <strong>Key drivers for this flight</strong>
                  <div style="height:8px"></div>
                  {driver_html}
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # ── What-if comparison ──────────────────────────────────────────────────
        st.markdown("**What-if: change weather severity**")
        severities  = [0, 1, 2]
        sev_labels  = ["0 — Clear", "1 — Mild", "2 — Multi"]
        sev_probs   = []
        whatif_backend_mode = "Proxy model" if backend_error else backend_mode
        for s in severities:
            sev_features = dict(features)
            sev_features["weather_severity"] = s
            sev_prob, _, _ = predict_probability(model, meta, sev_features, whatif_backend_mode)
            sev_probs.append(sev_prob)

        fig2, ax2 = plt.subplots(figsize=(8.2, 3.2))
        fig2.patch.set_facecolor("#ffffff")
        ax2.set_facecolor("#ffffff")
        colors_wi = ["#86c5a0" if s != weather_severity else "#2563eb" for s in severities]
        bars = ax2.bar(sev_labels, [p * 100 for p in sev_probs],
                       color=colors_wi, width=0.46, edgecolor="white")
        for bar, p in zip(bars, sev_probs):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f"{p*100:.1f}%", ha="center", fontsize=10,
                     fontweight="bold", color="#102033")
        ax2.set_ylabel("Delay probability (%)")
        ax2.set_title("Delay probability across weather severities\n(all other inputs fixed)",
                      fontsize=11, color="#102033", fontweight="bold")
        ax2.tick_params(colors="#102033")
        ax2.yaxis.label.set_color("#102033")
        ax2.spines[["top","right"]].set_visible(False)
        ax2.set_ylim(0, max(sev_probs)*100 * 1.25)
        ax2.axhline(hist_rate * 100, color="#6b7280", linestyle="--",
                    linewidth=1, label=f"{origin} historical avg")
        ax2.legend(fontsize=8)
        st.pyplot(fig2, width="stretch")
        plt.close()

# ─── TAB 2: EDA Insights ──────────────────────────────────────────────────────
with tab_eda:
    col_a, col_b = st.columns(2)

    # Hive query results — replayed as interactive charts
    with col_a:
        st.markdown("#### Worst airports by delay rate (Hive Q2)")
        airports_eda = {
            "ASE":30.81,"FLL":28.16,"BWI":27.87,"DFW":26.57,"MIA":26.04,
            "MCO":25.76,"DAL":25.51,"MDW":25.41,"DEN":25.19,"LAS":24.58,
        }
        df_ap = pd.DataFrame(airports_eda.items(), columns=["Airport","Delay %"])
        df_ap = df_ap.sort_values("Delay %")
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        fig3.patch.set_facecolor("#f7f9fc"); ax3.set_facecolor("#f7f9fc")
        ax3.barh(df_ap["Airport"], df_ap["Delay %"], color="#4C72B0", height=0.6)
        ax3.set_xlabel("% of flights delayed >15 min")
        ax3.set_title("Top-10 most disrupted airports", fontsize=10)
        ax3.spines[["top","right"]].set_visible(False)
        ax3.grid(axis="x", linestyle="--", alpha=0.4)
        st.pyplot(fig3)
        plt.close()

    with col_b:
        st.markdown("#### Carrier delay rates (Hive Q4)")
        carrier_eda = {
            "F9":28.45,"B6":27.55,"NK":25.65,"WN":24.52,"AA":24.33,
            "G4":22.77,"OH":20.36,"AS":20.09,"UA":19.39,"DL":17.92,
        }
        df_c = pd.DataFrame(carrier_eda.items(), columns=["Carrier","Delay %"])
        df_c = df_c.sort_values("Delay %")
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        fig4.patch.set_facecolor("#f7f9fc"); ax4.set_facecolor("#f7f9fc")
        colors_c = ["#55A868" if v < 22 else ("#DD8452" if v < 26 else "#C44E52")
                    for v in df_c["Delay %"]]
        ax4.barh(df_c["Carrier"], df_c["Delay %"], color=colors_c, height=0.6)
        ax4.set_xlabel("% of flights delayed >15 min")
        ax4.set_title("Carrier delay rates (best → worst)", fontsize=10)
        ax4.spines[["top","right"]].set_visible(False)
        ax4.grid(axis="x", linestyle="--", alpha=0.4)
        st.pyplot(fig4)
        plt.close()

    st.markdown("#### Year-over-year trend (Hive Q1)")
    yearly = pd.DataFrame({
        "Year": [2023, 2024, 2025],
        "Avg delay (min)": [12.29, 12.75, 13.66],
        "% delayed": [20.51, 20.76, 21.9],
        "Flights (M)": [6.53, 6.76, 6.67],
    })
    st.dataframe(yearly.set_index("Year"), width="stretch")
    st.caption("Source: Apache Hive query on HDFS-stored Parquet (20.6M rows, 2023–2025)")

# ─── TAB 3: Model Performance ─────────────────────────────────────────────────
with tab_model:
    col_r, col_s = st.columns(2)

    with col_r:
        st.markdown("#### Model metrics")
        metrics = pd.DataFrame({
            "Metric": ["ROC-AUC", "PR-AUC", "F1 Score", "Positive class rate"],
            "Value":  ["0.655",   "0.338",  "0.677",    "~21%"],
            "Notes":  [
                "Published benchmarks: 0.65–0.75",
                "Baseline (random): 0.21",
                "At optimal threshold",
                "Test set 2025",
            ],
        })
        st.dataframe(metrics.set_index("Metric"), width="stretch")

        st.markdown("#### Pipeline architecture")
        st.markdown("""
| Tool | Role |
|------|------|
| **HDFS** (2 data nodes) | Raw CSV + Parquet storage, 4.9 GB raw |
| **PySpark** | Ingest, crosswalk join, feature engineering |
| **Apache Hive** | External tables on Parquet, 7 EDA queries |
| **MLlib RF** | 100-tree Random Forest classifier |
| **This demo** | sklearn proxy, no Spark required |
        """)

    with col_s:
        st.markdown("#### Calibration finding")
        st.markdown("""
The Random Forest outputs **over-confident probabilities** at higher confidence levels
(a known property of tree ensembles). Platt scaling corrects this without affecting AUC.

In production, calibration would be applied before serving probabilities to end users.
        """)

        # Simple calibration illustration
        raw_x  = [0.35, 0.43, 0.52, 0.61]
        raw_y  = [0.12, 0.20, 0.31, 0.49]
        cal_y  = [0.18, 0.26, 0.38, 0.55]
        perf   = [0.35, 0.43, 0.52, 0.61]

        fig5, ax5 = plt.subplots(figsize=(5, 4))
        fig5.patch.set_facecolor("#f7f9fc"); ax5.set_facecolor("#f7f9fc")
        ax5.plot([0,1],[0,1],"k--",lw=1.5,label="Perfect")
        ax5.plot(raw_x, raw_y, "s-", color="#4C72B0", lw=2, markersize=7, label="RF raw")
        ax5.plot(raw_x, cal_y, "^-", color="#DD8452", lw=2, markersize=7, label="Platt-scaled")
        ax5.set_xlabel("Mean predicted probability")
        ax5.set_ylabel("Observed delay rate")
        ax5.set_title("Calibration (schematic)", fontsize=10)
        ax5.legend(fontsize=8)
        ax5.set_xlim(0,1); ax5.set_ylim(0,1)
        ax5.spines[["top","right"]].set_visible(False)
        st.pyplot(fig5)
        plt.close()

        st.info("The calibration gap is a finding, not a bug. It motivates Platt scaling as a production improvement.")

st.markdown("---")
st.caption(
    "Built on HDFS + PySpark + Apache Hive · 20.6M flight records (BTS 2023–2025) "
    "+ NOAA ISD weather · Random Forest (AUC 0.655)"
)
