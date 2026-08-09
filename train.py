from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "outputs" / "features.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    if not DATA.exists():
        raise FileNotFoundError("Run `python prepare_data.py` first.")

    df = pd.read_csv(DATA)

    drop_cols = [c for c in ["label", "source_experiment", "fault_location"] if c in df]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
        min_samples_leaf=2,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions, digits=3)
    print(report)

    (OUTPUT_DIR / "metrics.txt").write_text(
        "Random Forest classification report\n\n" + report,
        encoding="utf-8"
    )

    cm = confusion_matrix(y_test, predictions, labels=model.classes_)
    ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=model.classes_
    ).plot(xticks_rotation=25)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=180)
    plt.close()

    importance = pd.Series(
        model.feature_importances_, index=feature_cols
    ).sort_values(ascending=True)

    plt.figure(figsize=(8, 5))
    importance.plot(kind="barh")
    plt.xlabel("Random Forest feature importance")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=180)
    plt.close()

    joblib.dump(
        {"model": model, "feature_columns": feature_cols},
        MODEL_DIR / "random_forest.joblib"
    )

    print("Saved:", MODEL_DIR / "random_forest.joblib")


if __name__ == "__main__":
    main()
