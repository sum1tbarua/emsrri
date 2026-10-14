# 01 forward temporal validation
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


# ============================================================
# CORRECT FORWARD TEMPORAL VALIDATION
# 2024 development -> 2025 forward temporal evaluation
# Self-contained: does not depend on previous notebook variables
# ============================================================

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    brier_score_loss
)

from xgboost import XGBClassifier


# ============================================================
# 1. Paths
# ============================================================

BASE = str(ANALYSIS_ROOT)

DATA_2024 = f"{BASE}/datasets/nemsis_2024.parquet"
DATA_2025 = f"{BASE}/datasets/nemsis_2025.parquet"

OUT = f"{BASE}/revision_forward_validation_corrected"
os.makedirs(OUT, exist_ok=True)


# ============================================================
# 2. Load event-level datasets
# ============================================================

df24 = pd.read_parquet(DATA_2024)
df25 = pd.read_parquet(DATA_2025)

print("2024 shape:", df24.shape)
print("2025 shape:", df25.shape)


# ============================================================
# 3. EXACT final 32-predictor specification
# ============================================================

numeric_features = [
    "computed_EMSChuteTimeMin",
    "computed_EMSDispatchCenterTimeSec",
    "computed_EMSSceneResponseTimeMin",
    "computed_EMSSceneTimeMin",
    "computed_EMSSceneToPatientTimeMin",
    "computed_EMSSystemResponseTimeMin",
    "computed_EMSTotalCallTimeMin",
    "computed_EMSTransportTimeMin",
    "computed_ageinyear",
]

categorical_features = [
    "computed_NasemsoRegion",
    "computed_USCensusDivision",
    "computed_USCensusRegion",
    "computed_Urbanicity",
    "eArrest_05",
    "eDisposition_16",
    "eDisposition_17",
    "eDisposition_19",
    "eDisposition_21",
    "eDisposition_22",
    "eDisposition_28",
    "eDisposition_29",
    "eDisposition_30",
    "eDisposition_32",
    "eOutcome_01",
    "eOutcome_02",
    "ePayment_01",
    "ePayment_50",
    "eSituation_02",
    "eSituation_07",
    "eSituation_08",
    "eSituation_13",
    "eSituation_20",
]

predictors = numeric_features + categorical_features

assert len(numeric_features) == 9
assert len(categorical_features) == 23
assert len(predictors) == 32

missing24 = [c for c in predictors if c not in df24.columns]
missing25 = [c for c in predictors if c not in df25.columns]

assert not missing24, f"Missing 2024 predictors: {missing24}"
assert not missing25, f"Missing 2025 predictors: {missing25}"

print("\nPredictor specification:")
print("Numeric:", len(numeric_features))
print("Categorical:", len(categorical_features))
print("Total:", len(predictors))


# ============================================================
# 4. Recreate 2024 development split
#    70% train / 15% validation / 15% internal test
# ============================================================

train24, temp24 = train_test_split(
    df24,
    test_size=0.30,
    random_state=42,
    shuffle=True
)

val24, test24 = train_test_split(
    temp24,
    test_size=0.50,
    random_state=42,
    shuffle=True
)

print("\n2024 development split:")
print(f"Train:         {len(train24):,}")
print(f"Validation:    {len(val24):,}")
print(f"Internal test: {len(test24):,}")
print(f"2025 forward:  {len(df25):,}")


# ============================================================
# 5. Derive EMS-RRI cutoff from 2024 TRAINING ONLY
# ============================================================

rri_cutoff = float(
    train24["ems_rri_basic"].quantile(0.20)
)

print("\n2024 training-derived EMS-RRI cutoff:")
print(rri_cutoff)


def derive_target(df, cutoff):
    return (
        df["ems_rri_basic"].astype(float) <= cutoff
    ).astype(int)


y_train = derive_target(train24, rri_cutoff)
y_val   = derive_target(val24, rri_cutoff)
y_test  = derive_target(test24, rri_cutoff)
y_2025  = derive_target(df25, rri_cutoff)

print("\nLow-reliability prevalence:")
for name, y in [
    ("2024 train", y_train),
    ("2024 validation", y_val),
    ("2024 internal test", y_test),
    ("2025 forward temporal", y_2025),
]:
    print(
        f"{name:24s}: "
        f"{y.sum():,}/{len(y):,} "
        f"({100*y.mean():.2f}%)"
    )


# ============================================================
# 6. Predictor matrices
# ============================================================

X_train = train24[predictors].copy()
X_val   = val24[predictors].copy()
X_test  = test24[predictors].copy()
X_2025  = df25[predictors].copy()


# ============================================================
# 7. Explicitly coerce the 9 numeric variables to numeric
#    Important because parquet currently stores them as objects
# ============================================================

for col in numeric_features:
    X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
    X_val[col]   = pd.to_numeric(X_val[col], errors="coerce")
    X_test[col]  = pd.to_numeric(X_test[col], errors="coerce")
    X_2025[col]  = pd.to_numeric(X_2025[col], errors="coerce")


# ============================================================
# 8. Exact original preprocessing
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=True
            )
        ),
        (
            "scaler",
            RobustScaler(
                with_centering=True,
                with_scaling=True
            )
        )
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ============================================================
# 9. Exact original XGBoost hyperparameters
# ============================================================

classifier = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ]
)


# ============================================================
# 10. Train exclusively on 2024 training data
# ============================================================

print("\nTraining corrected 2024 development model...")

model.fit(X_train, y_train)

print("Training complete.")


# ============================================================
# 11. Select classification threshold using
#     2024 VALIDATION ONLY
# ============================================================

val_prob = model.predict_proba(X_val)[:, 1]

threshold_rows = []

for threshold in np.arange(0.01, 1.00, 0.01):

    pred = (val_prob >= threshold).astype(int)

    threshold_rows.append({
        "threshold": round(float(threshold), 2),
        "f1": f1_score(y_val, pred),
        "precision": precision_score(
            y_val, pred, zero_division=0
        ),
        "recall": recall_score(
            y_val, pred, zero_division=0
        )
    })

threshold_df = pd.DataFrame(threshold_rows)

best_idx = threshold_df["f1"].idxmax()
best_row = threshold_df.loc[best_idx]

best_threshold = float(best_row["threshold"])

print("\nBest validation threshold:")
print(best_row)


# ============================================================
# 12. Evaluation function
# ============================================================

def evaluate(name, X, y, fitted_model, threshold):

    prob = fitted_model.predict_proba(X)[:, 1]
    pred = (prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y,
        pred,
        labels=[0, 1]
    ).ravel()

    return {
        "dataset": name,
        "n": len(y),

        "accuracy":
            accuracy_score(y, pred),

        "balanced_accuracy":
            balanced_accuracy_score(y, pred),

        "precision":
            precision_score(
                y, pred, zero_division=0
            ),

        "recall":
            recall_score(
                y, pred, zero_division=0
            ),

        "f1":
            f1_score(
                y, pred, zero_division=0
            ),

        "roc_auc":
            roc_auc_score(y, prob),

        "pr_auc":
            average_precision_score(y, prob),

        "brier_score":
            brier_score_loss(y, prob),

        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),

        "fpr":
            fp / (fp + tn),

        "fnr":
            fn / (fn + tp)
    }


# ============================================================
# 13. Final evaluation
# ============================================================

results = [
    evaluate(
        "2024 validation",
        X_val,
        y_val,
        model,
        best_threshold
    ),

    evaluate(
        "2024 internal test",
        X_test,
        y_test,
        model,
        best_threshold
    ),

    evaluate(
        "2025 forward temporal",
        X_2025,
        y_2025,
        model,
        best_threshold
    )
]

results_df = pd.DataFrame(results)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

print("\nFINAL FORWARD-VALIDATION RESULTS:")
print(results_df.to_string(index=False))


# ============================================================
# 14. Save experiment
# ============================================================

results_df.to_csv(
    f"{OUT}/forward_validation_metrics.csv",
    index=False
)

threshold_df.to_csv(
    f"{OUT}/threshold_search.csv",
    index=False
)

joblib.dump(
    model,
    f"{OUT}/xgboost_2024_to_2025.joblib"
)

config = {
    "design":
        "2024 development -> 2025 forward temporal validation",

    "rri_cutoff_source":
        "20th percentile of 2024 training EMS-RRI",

    "rri_cutoff":
        rri_cutoff,

    "training_n":
        len(train24),

    "validation_n":
        len(val24),

    "internal_test_n":
        len(test24),

    "forward_validation_n":
        len(df25),

    "selected_classifier_threshold":
        best_threshold,

    "threshold_selection":
        "Maximum F1 on 2024 validation set",

    "numeric_predictor_count":
        len(numeric_features),

    "categorical_predictor_count":
        len(categorical_features),

    "total_predictor_count":
        len(predictors),

    "random_state":
        42,

    "xgboost_parameters": {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "objective": "binary:logistic",
        "eval_metric": "logloss"
    }
}

with open(
    f"{OUT}/forward_validation_configuration.json",
    "w"
) as f:
    json.dump(config, f, indent=2)

print("\nOutputs saved to:")
print(OUT)