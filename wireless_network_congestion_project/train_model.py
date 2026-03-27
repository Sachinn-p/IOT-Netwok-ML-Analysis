import json
import os
from pathlib import Path
from typing import Dict, Tuple

os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


MODEL_FEATURES = [
    "packet_rate",
    "allocated_bandwidth",
    "latency",
    "packet_loss_rate",
    "network_load",
]
TARGET_COLUMN = "congestion_level"


def prepare_train_test_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    required_cols = MODEL_FEATURES + [TARGET_COLUMN]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for training: {missing_cols}")

    X = df[MODEL_FEATURES]
    y = df[TARGET_COLUMN]

    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def build_models() -> Dict[str, object]:
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(max_iter=1000, random_state=42),
                ),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=42, n_jobs=1
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
    }


def evaluate_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Dict[str, Dict[str, object]]:
    models = build_models()
    results: Dict[str, Dict[str, object]] = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = float(accuracy_score(y_test, y_pred))

        results[model_name] = {
            "model": model,
            "accuracy": accuracy,
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(
                y_test, y_pred, digits=4, zero_division=0
            ),
        }

    return results


def select_best_model(results: Dict[str, Dict[str, object]]) -> str:
    return max(results, key=lambda name: results[name]["accuracy"])


def save_training_outputs(
    results: Dict[str, Dict[str, object]],
    best_model_name: str,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    best_model_bundle = {
        "model_name": best_model_name,
        "model": results[best_model_name]["model"],
        "features": MODEL_FEATURES,
        "target_column": TARGET_COLUMN,
        "classes": list(results[best_model_name]["model"].classes_),
    }

    model_path = output_dir / "best_model.joblib"
    joblib.dump(best_model_bundle, model_path)

    metrics_dict = {
        model_name: {
            "accuracy": metrics["accuracy"],
            "confusion_matrix": metrics["confusion_matrix"],
            "classification_report": metrics["classification_report"],
        }
        for model_name, metrics in results.items()
    }
    metrics_path = output_dir / "model_metrics.json"
    metrics_path.write_text(json.dumps(metrics_dict, indent=2), encoding="utf-8")

    return model_path


def train_and_select_model(
    df: pd.DataFrame, output_dir: Path
) -> Tuple[Dict[str, Dict[str, object]], str, Path]:
    X_train, X_test, y_train, y_test = prepare_train_test_data(df)
    results = evaluate_models(X_train, X_test, y_train, y_test)
    best_model_name = select_best_model(results)
    model_path = save_training_outputs(results, best_model_name, output_dir)
    return results, best_model_name, model_path
