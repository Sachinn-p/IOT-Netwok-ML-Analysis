from pathlib import Path
from typing import Dict, Optional

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_PATH = (
    PROJECT_ROOT
    / ".kagglehub-cache/datasets/programmer3/iot-network-traffic-dataset/versions/1/Iot_Network_data.csv"
)


def resolve_dataset_path(file_path: Optional[str] = None) -> Path:
    if file_path:
        candidate = Path(file_path).expanduser().resolve()
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Dataset not found at: {candidate}")

    if DEFAULT_DATASET_PATH.exists():
        return DEFAULT_DATASET_PATH

    candidates = list(PROJECT_ROOT.rglob("Iot_Network_data.csv"))
    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        "Could not locate Iot_Network_data.csv. "
        "Pass file_path explicitly or run load_iot_dataset.py first."
    )


def load_dataset(file_path: Optional[str] = None) -> pd.DataFrame:
    dataset_path = resolve_dataset_path(file_path)
    return pd.read_csv(dataset_path)


def identify_important_features(df: pd.DataFrame) -> Dict[str, str]:
    bandwidth_feature = (
        "allocated_bandwidth"
        if "allocated_bandwidth" in df.columns
        else "bandwidth_usage"
        if "bandwidth_usage" in df.columns
        else "Not found"
    )

    return {
        "packet_size": "packet_size",
        "bandwidth": bandwidth_feature,
        "packets_per_second": "Derived as packet_rate = 1 / transmission_time",
        "flow_duration": "transmission_time",
        "bytes_per_second": "Derived as bytes_per_second = packet_size / transmission_time",
    }


def print_data_understanding(df: pd.DataFrame) -> None:
    print("\n=== 1) DATA UNDERSTANDING ===")
    print(f"Dataset shape: {df.shape}")
    print("\nColumns:")
    for column in df.columns:
        print(f"- {column}")

    print("\nData types:")
    print(df.dtypes.to_string())

    print("\nImportant network features:")
    feature_map = identify_important_features(df)
    for name, mapped_value in feature_map.items():
        print(f"- {name}: {mapped_value}")
