"""
Local plotting entrypoint for the dedicated Python plotter container.

The Spark container should export small CSV inputs into /data/plot_inputs.
This script is intentionally separate from make_plots.py, which runs in Spark.
"""
from pathlib import Path

import pandas as pd
import plotly.express as px
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, precision_recall_curve, roc_curve


INPUT_DIR = Path("/data/plot_inputs")
OUTPUT_DIR = Path("/data/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def read_csv(name: str) -> pd.DataFrame:
    path = INPUT_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Export plot input CSVs from Spark before running plotter."
        )
    return pd.read_csv(path)


def save_plot(fig, name: str) -> None:
    fig.write_html(OUTPUT_DIR / f"{name}.html")
    fig.write_image(OUTPUT_DIR / f"{name}.png")


def main() -> None:
    delay = read_csv("delay_distribution.csv")
    fig = px.histogram(
        delay,
        x="dep_delay",
        color="season",
        nbins=60,
        title="Departure delay distribution by season",
    )
    save_plot(fig, "01_delay_distribution")

    delay_box = delay[delay["dep_delay"].between(-20, 120)].copy()
    season_order = ["winter", "spring", "summer", "fall"]
    fig = px.box(
        delay_box,
        x="season",
        y="dep_delay",
        color="season",
        category_orders={"season": season_order},
        title="Departure delay distribution by season (clipped -20 to 120 min)",
        labels={"season": "Season", "dep_delay": "Departure delay (minutes)"},
        points="outliers",
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_hline(y=15, line_dash="dot", line_color="red")
    save_plot(fig, "01_delay_distribution_v2")

    airport = read_csv("airport_fingerprint.csv")
    fig = px.density_heatmap(
        airport,
        x="crs_dep_hour",
        y="origin",
        z="delay_rate",
        histfunc="avg",
        title="Airport disruption fingerprint",
        color_continuous_scale="YlOrRd",
    )
    save_plot(fig, "02_airport_fingerprint")

    preds = read_csv("predictions_sample.csv")

    fpr, tpr, _ = roc_curve(preds["dep_del15"], preds["p_delay"])
    fig = px.line(
        x=fpr,
        y=tpr,
        labels={"x": "False positive rate", "y": "True positive rate"},
        title=f"ROC curve (AUC={auc(fpr, tpr):.3f})",
    )
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash"))
    save_plot(fig, "03a_roc")

    precision, recall, _ = precision_recall_curve(preds["dep_del15"], preds["p_delay"])
    fig = px.line(
        x=recall,
        y=precision,
        labels={"x": "Recall", "y": "Precision"},
        title=f"Precision-Recall curve (AUC={auc(recall, precision):.3f})",
    )
    save_plot(fig, "03b_pr")

    weather = read_csv("weather_severity.csv")
    weather["weather_severity"] = weather["weather_severity"].astype(int).astype(str)
    weather["delay_rate_pct"] = weather["delay_rate"] * 100
    fig = px.bar(
        weather,
        x="weather_severity",
        y="delay_rate_pct",
        text="n",
        title="Delay rate vs weather severity",
        labels={
            "weather_severity": "Weather severity score (0=clear, 4=extreme)",
            "delay_rate_pct": "Delay rate (%)",
            "n": "Flights",
        },
    )
    fig.update_traces(texttemplate="n=%{text:,}", textposition="outside")
    fig.update_xaxes(type="category")
    fig.update_layout(yaxis_range=[0, weather["delay_rate_pct"].max() * 1.15])
    save_plot(fig, "04_weather_severity")

    calibration = read_csv("calibration.csv")
    fig = px.line(
        calibration,
        x="p_mean",
        y="y_mean",
        markers=True,
        title="Calibration plot",
        labels={"p_mean": "Mean predicted probability", "y_mean": "Observed delay rate"},
    )
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash"))
    save_plot(fig, "05_calibration")

    y_true = preds["dep_del15"].astype(int).to_numpy()
    y_prob = preds["p_delay"].astype(float).to_numpy()
    frac_pos_raw, mean_pred_raw = calibration_curve(y_true, y_prob, n_bins=10)

    platt = LogisticRegression(max_iter=1000)
    platt.fit(y_prob.reshape(-1, 1), y_true)
    y_prob_cal = platt.predict_proba(y_prob.reshape(-1, 1))[:, 1]
    frac_pos_cal, mean_pred_cal = calibration_curve(y_true, y_prob_cal, n_bins=10)

    cal_v2 = pd.concat(
        [
            pd.DataFrame({
                "mean_predicted_probability": mean_pred_raw,
                "observed_delay_rate": frac_pos_raw,
                "model": "Random Forest (raw)",
            }),
            pd.DataFrame({
                "mean_predicted_probability": mean_pred_cal,
                "observed_delay_rate": frac_pos_cal,
                "model": "Platt-scaled RF",
            }),
        ],
        ignore_index=True,
    )
    fig = px.line(
        cal_v2,
        x="mean_predicted_probability",
        y="observed_delay_rate",
        color="model",
        markers=True,
        title="Calibration plot — RF vs Platt-scaled RF",
        labels={
            "mean_predicted_probability": "Mean predicted probability",
            "observed_delay_rate": "Observed delay rate",
            "model": "Model",
        },
    )
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash"))
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    save_plot(fig, "05_calibration_v2")

    feature_importance_path = INPUT_DIR / "feature_importance.csv"
    if feature_importance_path.exists():
        feature_importance = read_csv("feature_importance.csv").head(15)
        feature_importance = feature_importance.sort_values("importance", ascending=True)
        fig = px.bar(
            feature_importance,
            x="importance",
            y="feature",
            color="category",
            orientation="h",
            title="Top-15 feature importances — Random Forest classifier",
            labels={
                "importance": "Feature importance (mean decrease in impurity)",
                "feature": "Feature",
                "category": "Category",
            },
            color_discrete_map={
                "weather": "#4C72B0",
                "temporal/flight": "#DD8452",
                "carrier": "#55A868",
                "airport": "#C44E52",
                "other": "#937860",
            },
        )
        fig.update_layout(
            yaxis={
                "categoryorder": "array",
                "categoryarray": feature_importance["feature"].tolist(),
            }
        )
        save_plot(fig, "06_feature_importance")
    else:
        print("Skipping 06_feature_importance: feature_importance.csv was not exported.")

    print(f"Plots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
