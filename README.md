# Induction Motor Predictive Maintenance

End-to-end condition-monitoring project using the Paderborn University Bearing Dataset.

Pipeline:
Paderborn data -> motor current + vibration -> RMS/variance/peak-to-peak/kurtosis
-> concatenate features -> Random Forest -> Healthy/Inner Race/Outer Race
-> Gradio dashboard.

This is intentionally a simple, explainable time-domain baseline: no FFT and no deep learning.

## Dataset

Official Paderborn University Bearing DataCenter:
https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter/data-sets-and-download

The project uses the public `paderborn-bearing` preprocessing package:
https://github.com/JvdHoogen/paderborn_bearing

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
# Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Run

1. Prepare features:

```bash
python prepare_data.py
```

2. Train Random Forest:

```bash
python train.py
```

3. Launch dashboard:

```bash
python app.py
```

Outputs:
- `outputs/features.csv`
- `models/random_forest.joblib`
- `outputs/metrics.txt`
- `outputs/confusion_matrix.png`
- `outputs/feature_importance.png`

The raw Paderborn dataset is NOT included because it is large and separately licensed.

## Project scope

Healthy + artificial inner-race + artificial outer-race bearing conditions.

The model is a condition-monitoring/fault-classification system. It does not predict remaining useful life or time-to-failure.

## Interview explanation

Vibration captures the mechanical behavior of the bearing. Motor current provides an electrical signature influenced by the mechanical condition. For each synchronized window, statistical features are extracted independently and concatenated into one tabular feature vector. Random Forest then classifies the bearing condition.

Why Random Forest? The resulting dataset is small, tabular and nonlinear, and Random Forest requires no feature scaling while providing feature importance.
