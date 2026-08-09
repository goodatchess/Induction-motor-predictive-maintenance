from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from features import build_feature_table

# Start with three easy-to-explain classes.
EXPERIMENTS = [
    ("Healthy", "Normal"),
    ("Artificial", "IR"),
    ("Artificial", "OR"),
]

WINDOW_LENGTH = 2048
MAX_SAMPLES_PER_EXPERIMENT = 600
OUTPUT = ROOT / "outputs" / "features.csv"


def load_experiment(experiment, fault_location):
    from paderborn_bearing import Paderborn

    print(f"Loading: {experiment} / {fault_location}")
    data = Paderborn(experiment, WINDOW_LENGTH, fault_location)

    current = np.asarray(data.motor_current)
    vibration = np.asarray(data.vibrations)
    labels = np.asarray(data.labels).ravel()

    n = min(len(labels), MAX_SAMPLES_PER_EXPERIMENT)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(labels), size=n, replace=False)

    return current[idx], vibration[idx], labels[idx]


def normalize_label(label):
    text = str(label).upper()

    if "NORMAL" in text or text in {"0", "HEALTHY", "H"}:
        return "Healthy"
    if "IR" in text or "INNER" in text:
        return "Inner Race Fault"
    if "OR" in text or "OUTER" in text:
        return "Outer Race Fault"

    return str(label)


def main():
    tables = []

    for experiment, location in EXPERIMENTS:
        current, vibration, labels = load_experiment(experiment, location)
        table = build_feature_table(current, vibration, labels)
        table["label"] = table["label"].map(normalize_label)
        tables.append(table)

    df = pd.concat(tables, ignore_index=True).dropna()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)

    print("\nSaved:", OUTPUT)
    print("\nClass distribution:")
    print(df["label"].value_counts())
    print("\nShape:", df.shape)


if __name__ == "__main__":
    main()
