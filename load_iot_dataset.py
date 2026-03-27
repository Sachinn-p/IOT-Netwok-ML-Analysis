import os
from pathlib import Path

import kagglehub
from kagglehub import KaggleDatasetAdapter


DATASET_REF = "programmer3/iot-network-traffic-dataset"

# Set a file path explicitly with IOT_FILE_PATH if desired.
file_path = os.environ.get("IOT_FILE_PATH", "").strip()

if not file_path:
    dataset_dir = Path(kagglehub.dataset_download(DATASET_REF))
    csv_files = sorted(p for p in dataset_dir.rglob("*.csv") if p.is_file())
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under: {dataset_dir}")
    file_path = str(csv_files[0].relative_to(dataset_dir))
    print(f"Using discovered file_path: {file_path}")

df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    DATASET_REF,
    file_path,
)

print("First 5 records:")
print(df.head())
