# 11 sgd baseline
# See companion notebook and docs/PROVENANCE.md.
# Repository configuration; no data are loaded by this cell.
import os
from pathlib import Path
if "EMS_RRI_REPOSITORY_ROOT" in os.environ:
    REPOSITORY_ROOT = Path(os.environ["EMS_RRI_REPOSITORY_ROOT"]).expanduser().resolve()
elif "__file__" in globals():
    REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
else:
    REPOSITORY_ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
                            if (p / "reference/scoring_rules.json").exists()), None)
    if REPOSITORY_ROOT is None:
        raise RuntimeError("Start Jupyter inside the repository, or set EMS_RRI_REPOSITORY_ROOT.")
ANALYSIS_ROOT = Path(os.environ.get("EMS_RRI_ANALYSIS_ROOT", str(REPOSITORY_ROOT / "work"))).expanduser().resolve()
ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
print("Analysis data/output root:", ANALYSIS_ROOT)


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, brier_score_loss
)

base = str(ANALYSIS_ROOT / "datasets")

numeric = [
    "computed_EMSChuteTimeMin", "computed_EMSDispatchCenterTimeSec",
    "computed_EMSSceneResponseTimeMin", "computed_EMSSceneTimeMin",
    "computed_EMSSceneToPatientTimeMin", "computed_EMSSystemResponseTimeMin",
    "computed_EMSTotalCallTimeMin", "computed_EMSTransportTimeMin",
    "computed_ageinyear",
]
categorical = [
    "computed_NasemsoRegion", "computed_USCensusDivision",
    "computed_USCensusRegion", "computed_Urbanicity", "eArrest_05",
    "eDisposition_16", "eDisposition_17", "eDisposition_19",
    "eDisposition_21", "eDisposition_22", "eDisposition_28",
    "eDisposition_29", "eDisposition_30", "eDisposition_32",
    "eOutcome_01", "eOutcome_02", "ePayment_01", "ePayment_50",
    "eSituation_02", "eSituation_07", "eSituation_08",
    "eSituation_13", "eSituation_20",
    "primary_impression_eSituation_11",
    "primary_symptom_eSituation_09",
]
predictors = numeric + categorical
assert len(predictors) == 34

columns = predictors + ["ems_rri_basic"]
df24 = pd.read_parquet(f"{base}/nemsis_2024.parquet", columns=columns)
df25 = pd.read_parquet(f"{base}/nemsis_2025.parquet", columns=columns)
assert (len(df24), len(df25)) == (200_000, 500_000)

train24, temp24 = train_test_split(
    df24, test_size=0.30, random_state=42, shuffle=True
)
val24, test24 = train_test_split(
    temp24, test_size=0.50, random_state=42, shuffle=True
)
assert (len(train24), len(val24), len(test24)) == (140_000, 30_000, 30_000)

cutoff = float(train24["ems_rri_basic"].quantile(0.20))
assert np.isclose(cutoff, 81.8182), f"Unexpected cutoff: {cutoff}"

def features(frame):
    x = frame[predictors].copy()
    for name in numeric:
        x[name] = pd.to_numeric(x[name], errors="coerce")
    return x

def target(frame):
    scores = pd.to_numeric(frame["ems_rri_basic"], errors="coerce")
    return (scores <= cutoff).astype(int).to_numpy()

preprocessor = ColumnTransformer([
    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", RobustScaler()),
    ]), numeric),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ]), categorical),
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", SGDClassifier(
        loss="log_loss", penalty="l2", alpha=0.0001,
        class_weight="balanced", max_iter=1000, tol=1e-3,
        random_state=42,
    )),
])

model.fit(features(train24), target(train24))

y_val = target(val24)
p_val = model.predict_proba(features(val24))[:, 1]
grid = np.arange(0.01, 1.00, 0.01)
threshold = float(grid[np.argmax([
    f1_score(y_val, p_val >= t) for t in grid
])])

rows = []
for name, frame, probability in [
    ("2024 validation", val24, p_val),
    ("2024 internal test", test24,
     model.predict_proba(features(test24))[:, 1]),
    ("2025 forward temporal", df25,
     model.predict_proba(features(df25))[:, 1]),
]:
    y = target(frame)
    predicted = probability >= threshold
    rows.append({
        "dataset": name,
        "n": len(frame),
        "threshold": threshold,
        "roc_auc": roc_auc_score(y, probability),
        "pr_auc": average_precision_score(y, probability),
        "f1": f1_score(y, predicted),
        "precision": precision_score(y, predicted),
        "recall": recall_score(y, predicted),
        "brier": brier_score_loss(y, probability),
    })

print(f"2024 training cutoff: {cutoff:.4f}")
print(f"SGD threshold selected on 2024 validation: {threshold:.2f}")
print(pd.DataFrame(rows).round(6).to_string(index=False))