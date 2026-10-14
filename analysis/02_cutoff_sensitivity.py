# 02 cutoff sensitivity
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
# EMS-RRI REVISION ANALYSIS 02
# Cutoff Sensitivity Analysis
#
# Purpose:
# Test whether the low-reliability prediction findings depend
# strongly on the originally selected 20th-percentile EMS-RRI
# cutoff.
#
# Design:
#   Development: 2024 NEMSIS
#       70% train
#       15% validation
#       15% internal test
#
#   Forward temporal validation:
#       Full 2025 NEMSIS cohort
#
# For each candidate percentile:
#   1. Derive EMS-RRI cutoff from 2024 TRAINING ONLY
#   2. Define low-reliability labels using that cutoff
#   3. Train same 32-predictor XGBoost model
#   4. Select classifier threshold on 2024 validation by max F1
#   5. Evaluate 2024 internal test
#   6. Evaluate full 2025 forward temporal cohort
# ============================================================


# ============================================================
# 1. Imports
# ============================================================

import os
import json
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
# 2. Paths
# ============================================================

BASE = str(ANALYSIS_ROOT)

DATA_2024 = f"{BASE}/datasets/nemsis_2024.parquet"
DATA_2025 = f"{BASE}/datasets/nemsis_2025.parquet"

OUT = f"{BASE}/revision_analysis/outputs/02_cutoff_sensitivity"
os.makedirs(OUT, exist_ok=True)

print("Output directory:")
print(OUT)


# ============================================================
# 3. Load datasets
# ============================================================

df24 = pd.read_parquet(DATA_2024)
df25 = pd.read_parquet(DATA_2025)

print("\nDataset shapes:")
print("2024:", df24.shape)
print("2025:", df25.shape)


# ============================================================
# 4. Exact final 32-predictor specification
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

for col in predictors + ["ems_rri_basic"]:
    assert col in df24.columns, f"Missing from 2024: {col}"
    assert col in df25.columns, f"Missing from 2025: {col}"

print("\nPredictors verified:")
print("Numeric:", len(numeric_features))
print("Categorical:", len(categorical_features))
print("Total:", len(predictors))


# ============================================================
# 5. Reproduce 2024 development split
#    EXACTLY as forward-validation experiment
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

print("\n2024 splits:")
print(f"Train:         {len(train24):,}")
print(f"Validation:    {len(val24):,}")
print(f"Internal test: {len(test24):,}")
print(f"2025 forward:  {len(df25):,}")


# ============================================================
# 6. Predictor matrices
# ============================================================

X_train = train24[predictors].copy()
X_val   = val24[predictors].copy()
X_test  = test24[predictors].copy()
X_2025  = df25[predictors].copy()

# Restore numeric fields explicitly
for col in numeric_features:
    X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
    X_val[col]   = pd.to_numeric(X_val[col], errors="coerce")
    X_test[col]  = pd.to_numeric(X_test[col], errors="coerce")
    X_2025[col]  = pd.to_numeric(X_2025[col], errors="coerce")


# ============================================================
# 7. Candidate EMS-RRI percentiles
# ============================================================

candidate_percentiles = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30
]

cutoff_mapping = []

for q in candidate_percentiles:

    cutoff = float(
        train24["ems_rri_basic"].quantile(q)
    )

    cutoff_mapping.append({
        "percentile": int(q * 100),
        "quantile": q,
        "ems_rri_cutoff": cutoff
    })

cutoff_mapping_df = pd.DataFrame(cutoff_mapping)

print("\nPercentile-to-cutoff mapping:")
print(cutoff_mapping_df.to_string(index=False))


# ============================================================
# 8. Identify duplicate effective cutoffs
#
# EMS-RRI is discrete, so multiple percentiles may map
# to exactly the same score threshold.
# ============================================================

cutoff_groups = (
    cutoff_mapping_df
    .groupby("ems_rri_cutoff")["percentile"]
    .apply(list)
    .reset_index()
)

print("\nUnique effective cutoffs:")
print(cutoff_groups.to_string(index=False))


# ============================================================
# 9. Preprocessing/model construction function
# ============================================================

def build_model():

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

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ]
    )


# ============================================================
# 10. Helper functions
# ============================================================

def make_target(df, cutoff):

    return (
        pd.to_numeric(
            df["ems_rri_basic"],
            errors="coerce"
        ) <= cutoff
    ).astype(int)


def find_best_f1_threshold(y_true, probabilities):

    rows = []

    # Same 0.01-grid procedure used previously
    for threshold in np.arange(0.01, 1.00, 0.01):

        pred = (
            probabilities >= threshold
        ).astype(int)

        rows.append({
            "classifier_threshold":
                round(float(threshold), 2),

            "f1":
                f1_score(
                    y_true,
                    pred,
                    zero_division=0
                ),

            "precision":
                precision_score(
                    y_true,
                    pred,
                    zero_division=0
                ),

            "recall":
                recall_score(
                    y_true,
                    pred,
                    zero_division=0
                )
        })

    threshold_df = pd.DataFrame(rows)

    best_idx = threshold_df["f1"].idxmax()

    return (
        float(
            threshold_df.loc[
                best_idx,
                "classifier_threshold"
            ]
        ),
        threshold_df
    )


def evaluate(
    dataset_name,
    y_true,
    probabilities,
    classifier_threshold
):

    pred = (
        probabilities >= classifier_threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        pred,
        labels=[0, 1]
    ).ravel()

    return {
        "dataset": dataset_name,
        "n": len(y_true),

        "low_reliability_n":
            int(y_true.sum()),

        "low_reliability_prevalence":
            float(y_true.mean()),

        "accuracy":
            accuracy_score(
                y_true,
                pred
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                y_true,
                pred
            ),

        "precision":
            precision_score(
                y_true,
                pred,
                zero_division=0
            ),

        "recall":
            recall_score(
                y_true,
                pred,
                zero_division=0
            ),

        "f1":
            f1_score(
                y_true,
                pred,
                zero_division=0
            ),

        "roc_auc":
            roc_auc_score(
                y_true,
                probabilities
            ),

        "pr_auc":
            average_precision_score(
                y_true,
                probabilities
            ),

        "brier_score":
            brier_score_loss(
                y_true,
                probabilities
            ),

        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),

        "fpr":
            float(
                fp / (fp + tn)
            ),

        "fnr":
            float(
                fn / (fn + tp)
            )
    }


# ============================================================
# 11. Run sensitivity analysis
# ============================================================

all_results = []
all_threshold_results = []

unique_cutoffs = sorted(
    cutoff_mapping_df[
        "ems_rri_cutoff"
    ].unique()
)

print(
    f"\nNumber of requested percentiles: "
    f"{len(candidate_percentiles)}"
)

print(
    f"Number of unique EMS-RRI cutoffs: "
    f"{len(unique_cutoffs)}"
)


for i, rri_cutoff in enumerate(
    unique_cutoffs,
    start=1
):

    mapped_percentiles = (
        cutoff_mapping_df.loc[
            cutoff_mapping_df[
                "ems_rri_cutoff"
            ] == rri_cutoff,
            "percentile"
        ]
        .tolist()
    )

    percentile_label = ",".join(
        str(x)
        for x in mapped_percentiles
    )

    print("\n" + "=" * 70)
    print(
        f"RUN {i}/{len(unique_cutoffs)}"
    )
    print(
        "EMS-RRI cutoff:",
        rri_cutoff
    )
    print(
        "Mapped percentile(s):",
        mapped_percentiles
    )
    print("=" * 70)


    # --------------------------------------------------------
    # Targets
    # --------------------------------------------------------

    y_train = make_target(
        train24,
        rri_cutoff
    )

    y_val = make_target(
        val24,
        rri_cutoff
    )

    y_test = make_target(
        test24,
        rri_cutoff
    )

    y_2025 = make_target(
        df25,
        rri_cutoff
    )


    print("\nLow-reliability prevalence:")

    for name, y in [
        ("2024 train", y_train),
        ("2024 validation", y_val),
        ("2024 internal test", y_test),
        ("2025 forward", y_2025)
    ]:

        print(
            f"{name:22s}: "
            f"{y.sum():,}/{len(y):,} "
            f"({100*y.mean():.2f}%)"
        )


    # --------------------------------------------------------
    # Train fresh model
    # --------------------------------------------------------

    model = build_model()

    print("\nTraining...")

    model.fit(
        X_train,
        y_train
    )

    print("Training complete.")


    # --------------------------------------------------------
    # Validation threshold selection
    # --------------------------------------------------------

    val_prob = model.predict_proba(
        X_val
    )[:, 1]

    best_threshold, threshold_df = (
        find_best_f1_threshold(
            y_val,
            val_prob
        )
    )

    print(
        "Selected classifier threshold:",
        best_threshold
    )


    threshold_df[
        "ems_rri_cutoff"
    ] = rri_cutoff

    threshold_df[
        "mapped_percentiles"
    ] = percentile_label

    all_threshold_results.append(
        threshold_df
    )


    # --------------------------------------------------------
    # Probabilities
    # --------------------------------------------------------

    test_prob = model.predict_proba(
        X_test
    )[:, 1]

    forward_prob = model.predict_proba(
        X_2025
    )[:, 1]


    # --------------------------------------------------------
    # Evaluate internal test
    # --------------------------------------------------------

    internal_metrics = evaluate(
        "2024 internal test",
        y_test,
        test_prob,
        best_threshold
    )

    internal_metrics[
        "ems_rri_cutoff"
    ] = rri_cutoff

    internal_metrics[
        "mapped_percentiles"
    ] = percentile_label

    internal_metrics[
        "classifier_threshold"
    ] = best_threshold

    all_results.append(
        internal_metrics
    )


    # --------------------------------------------------------
    # Evaluate 2025 forward temporal validation
    # --------------------------------------------------------

    forward_metrics = evaluate(
        "2025 forward temporal",
        y_2025,
        forward_prob,
        best_threshold
    )

    forward_metrics[
        "ems_rri_cutoff"
    ] = rri_cutoff

    forward_metrics[
        "mapped_percentiles"
    ] = percentile_label

    forward_metrics[
        "classifier_threshold"
    ] = best_threshold

    all_results.append(
        forward_metrics
    )


    print("\n2024 internal test:")
    print(
        f"ROC-AUC={internal_metrics['roc_auc']:.4f}, "
        f"PR-AUC={internal_metrics['pr_auc']:.4f}, "
        f"F1={internal_metrics['f1']:.4f}, "
        f"Recall={internal_metrics['recall']:.4f}, "
        f"FNR={internal_metrics['fnr']:.4f}"
    )

    print("2025 forward temporal:")
    print(
        f"ROC-AUC={forward_metrics['roc_auc']:.4f}, "
        f"PR-AUC={forward_metrics['pr_auc']:.4f}, "
        f"F1={forward_metrics['f1']:.4f}, "
        f"Recall={forward_metrics['recall']:.4f}, "
        f"FNR={forward_metrics['fnr']:.4f}"
    )


# ============================================================
# 12. Combine results
# ============================================================

results_df = pd.DataFrame(
    all_results
)

threshold_search_df = pd.concat(
    all_threshold_results,
    ignore_index=True
)

# Put important columns first
front_cols = [
    "mapped_percentiles",
    "ems_rri_cutoff",
    "classifier_threshold",
    "dataset",
    "n",
    "low_reliability_n",
    "low_reliability_prevalence",
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "brier_score",
    "fpr",
    "fnr",
    "tn",
    "fp",
    "fn",
    "tp"
]

results_df = results_df[
    front_cols
]

pd.set_option(
    "display.max_columns",
    None
)

pd.set_option(
    "display.width",
    220
)


# ============================================================
# 13. Print full sensitivity table
# ============================================================

print("\n\n")
print("=" * 90)
print("CUTOFF SENSITIVITY RESULTS")
print("=" * 90)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# 14. Compact comparison table
# ============================================================

compact = results_df[
    [
        "mapped_percentiles",
        "ems_rri_cutoff",
        "classifier_threshold",
        "dataset",
        "low_reliability_prevalence",
        "roc_auc",
        "pr_auc",
        "f1",
        "precision",
        "recall",
        "fnr",
        "brier_score"
    ]
].copy()

compact[
    "low_reliability_prevalence"
] *= 100

compact = compact.rename(
    columns={
        "mapped_percentiles":
            "Percentile(s)",

        "ems_rri_cutoff":
            "EMS-RRI cutoff",

        "classifier_threshold":
            "Model threshold",

        "low_reliability_prevalence":
            "Low-reliability %",

        "roc_auc":
            "ROC-AUC",

        "pr_auc":
            "PR-AUC",

        "f1":
            "F1",

        "precision":
            "Precision",

        "recall":
            "Recall",

        "fnr":
            "FNR",

        "brier_score":
            "Brier"
    }
)

print("\nCompact comparison:")
print(
    compact.round(4).to_string(
        index=False
    )
)


# ============================================================
# 15. Save outputs
# ============================================================

cutoff_mapping_df.to_csv(
    f"{OUT}/percentile_cutoff_mapping.csv",
    index=False
)

results_df.to_csv(
    f"{OUT}/cutoff_sensitivity_full_results.csv",
    index=False
)

compact.to_csv(
    f"{OUT}/cutoff_sensitivity_compact.csv",
    index=False
)

threshold_search_df.to_csv(
    f"{OUT}/classifier_threshold_searches.csv",
    index=False
)


# ============================================================
# 16. Save analysis configuration
# ============================================================

configuration = {

    "analysis":
        "EMS-RRI cutoff sensitivity",

    "development_year":
        2024,

    "forward_validation_year":
        2025,

    "candidate_percentiles":
        [
            int(x * 100)
            for x in candidate_percentiles
        ],

    "cutoff_source":
        "2024 training set only",

    "training_n":
        len(train24),

    "validation_n":
        len(val24),

    "internal_test_n":
        len(test24),

    "forward_validation_n":
        len(df25),

    "predictor_count":
        len(predictors),

    "numeric_predictor_count":
        len(numeric_features),

    "categorical_predictor_count":
        len(categorical_features),

    "model_threshold_selection":
        "Maximum F1 on 2024 validation set",

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
    f"{OUT}/cutoff_sensitivity_configuration.json",
    "w"
) as f:

    json.dump(
        configuration,
        f,
        indent=2
    )


print("\nSaved outputs to:")
print(OUT)