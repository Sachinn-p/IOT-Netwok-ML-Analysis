from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")

matplotlib.use("Agg")


def clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    cleaned_df = df.copy()
    missing_before = int(cleaned_df.isna().sum().sum())
    duplicate_count = int(cleaned_df.duplicated().sum())

    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    categorical_cols = cleaned_df.select_dtypes(exclude=[np.number]).columns

    for col in numeric_cols:
        cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())

    for col in categorical_cols:
        mode = cleaned_df[col].mode(dropna=True)
        fallback = "Unknown"
        cleaned_df[col] = cleaned_df[col].fillna(
            mode.iloc[0] if not mode.empty else fallback
        )

    cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
    missing_after = int(cleaned_df.isna().sum().sum())

    report = {
        "missing_values_before": missing_before,
        "missing_values_after": missing_after,
        "duplicates_removed": duplicate_count,
    }
    return cleaned_df, report


def _min_max_series(series: pd.Series) -> pd.Series:
    numeric_series = series.astype(float)
    min_value = numeric_series.min()
    max_value = numeric_series.max()

    if pd.isna(min_value) or pd.isna(max_value) or max_value == min_value:
        return pd.Series(0.0, index=numeric_series.index)

    return (numeric_series - min_value) / (max_value - min_value)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    featured_df = df.copy()

    transmission_time = featured_df["transmission_time"].clip(lower=1e-6)
    bandwidth = (
        featured_df["allocated_bandwidth"]
        if "allocated_bandwidth" in featured_df.columns
        else featured_df["bandwidth_usage"]
    )

    featured_df["flow_duration"] = transmission_time
    featured_df["packet_rate"] = 1.0 / transmission_time
    featured_df["bytes_per_second"] = featured_df["packet_size"] / transmission_time
    featured_df["network_load"] = featured_df["packet_rate"] * bandwidth
    bandwidth_usage = featured_df.get(
        "bandwidth_usage", pd.Series(0.0, index=featured_df.index)
    )
    bandwidth_pressure = bandwidth_usage / bandwidth.clip(lower=1e-6)

    protocol_risk = featured_df.get(
        "protocol_type", pd.Series("Unknown", index=featured_df.index)
    ).map({"TCP": 0.00, "HTTP": 0.03, "UDP": 0.05}).fillna(0.01)
    device_risk = featured_df.get(
        "device_type", pd.Series("Unknown", index=featured_df.index)
    ).map({"sensor": 0.00, "actuator": 0.02, "camera": 0.04}).fillna(0.01)

    congestion_score = (
        0.24 * _min_max_series(featured_df["packet_rate"])
        + 0.18 * _min_max_series(featured_df["latency"])
        + 0.18 * _min_max_series(featured_df["packet_loss_rate"])
        + 0.14
        * _min_max_series(
            featured_df.get("jitter", pd.Series(0.0, index=featured_df.index))
        )
        + 0.16 * _min_max_series(bandwidth_pressure.clip(upper=10))
        + 0.10
        * _min_max_series(
            featured_df.get("energy_usage", pd.Series(0.0, index=featured_df.index))
        )
        + protocol_risk
        + device_risk
    )
    featured_df["congestion_score"] = congestion_score

    q1 = congestion_score.quantile(0.33)
    q2 = congestion_score.quantile(0.66)

    featured_df["congestion_level"] = pd.cut(
        congestion_score,
        bins=[-np.inf, q1, q2, np.inf],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    ).astype(str)

    return featured_df


def normalize_numeric_features(
    df: pd.DataFrame, exclude_columns: Optional[Iterable[str]] = None
) -> Tuple[pd.DataFrame, MinMaxScaler]:
    normalized_df = df.copy()
    exclude_set = set(exclude_columns or [])
    numeric_cols = [
        col
        for col in normalized_df.select_dtypes(include=[np.number]).columns
        if col not in exclude_set
    ]

    scaler = MinMaxScaler()
    if numeric_cols:
        normalized_df[numeric_cols] = scaler.fit_transform(normalized_df[numeric_cols])
    return normalized_df, scaler


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Heatmap")
    fig.tight_layout()

    out_path = output_dir / "correlation_heatmap.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_bandwidth_vs_packet_rate(df: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_df = df.sample(n=min(5000, len(df)), random_state=42)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=sample_df,
        x="allocated_bandwidth",
        y="packet_rate",
        hue="congestion_level",
        alpha=0.6,
        ax=ax,
    )
    ax.set_title("Bandwidth vs Packet Rate")
    fig.tight_layout()

    out_path = output_dir / "bandwidth_vs_packet_rate.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_congestion_distribution(df: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    order = ["Low", "Medium", "High"]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df, x="congestion_level", order=order, ax=ax)
    ax.set_title("Congestion Level Distribution")
    ax.set_xlabel("Congestion Level")
    ax.set_ylabel("Count")
    fig.tight_layout()

    out_path = output_dir / "congestion_distribution.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_congestion_trends(df: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    trend_df = df.copy().reset_index(drop=True)
    trend_df["sample_index"] = trend_df.index
    trend_df["rolling_network_load"] = trend_df["network_load"].rolling(500).mean()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        trend_df["sample_index"],
        trend_df["rolling_network_load"],
        color="#1f77b4",
        linewidth=1.5,
    )
    ax.set_title("Network Congestion Trend (Rolling Network Load)")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Rolling Network Load (window=500)")
    fig.tight_layout()

    out_path = output_dir / "congestion_trend.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def run_eda(df: pd.DataFrame, output_dir: Path) -> Dict[str, Path]:
    return {
        "correlation_heatmap": plot_correlation_heatmap(df, output_dir),
        "bandwidth_vs_packet_rate": plot_bandwidth_vs_packet_rate(df, output_dir),
        "congestion_distribution": plot_congestion_distribution(df, output_dir),
        "congestion_trend": plot_congestion_trends(df, output_dir),
    }
