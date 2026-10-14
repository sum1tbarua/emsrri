# 03 boundary three tier
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
# EMS-RRI REVISION ANALYSIS 03
# Boundary Errors, High-Sensitivity Operating Points,
# and Three-Tier Reliability Screening
#
# Design:
#   Development year: 2024
#   Forward validation: 2025
#
# Uses:
#   EMS-RRI cutoff = derived from 2024 training only
#   Same final 32-predictor XGBoost pipeline
#
# Goals:
#   1. Compare F1-optimal threshold with higher-sensitivity
#      operating points.
#   2. Quantify false-negative severity near the EMS-RRI cutoff.
#   3. Build a three-tier scheme:
#         Reliable
#         Borderline / Uncertain
#         Low Reliability
#
#   Thresholds for the three-tier scheme are selected only
#   from the 2024 validation set.
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

OUT = f"{BASE}/revision_analysis/outputs/03_boundary_three_tier"
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
# 4. Exact 32-predictor specification
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

assert len(predictors) == 32


# ============================================================
# 5. Recreate 2024 development split
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

print("\n2024 split:")
print(f"Train:         {len(train24):,}")
print(f"Validation:    {len(val24):,}")
print(f"Internal test: {len(test24):,}")
print(f"2025 forward:  {len(df25):,}")


# ============================================================
# 6. Derive EMS-RRI cutoff from 2024 TRAIN ONLY
# ============================================================

rri_cutoff = float(
    train24["ems_rri_basic"].quantile(0.20)
)

print("\n2024 training-derived EMS-RRI cutoff:")
print(rri_cutoff)


def make_target(df, cutoff):
    rri = pd.to_numeric(
        df["ems_rri_basic"],
        errors="coerce"
    )

    return (rri <= cutoff).astype(int)


y_train = make_target(train24, rri_cutoff)
y_val   = make_target(val24, rri_cutoff)
y_test  = make_target(test24, rri_cutoff)
y_2025  = make_target(df25, rri_cutoff)


# ============================================================
# 7. Predictor matrices
# ============================================================

X_train = train24[predictors].copy()
X_val   = val24[predictors].copy()
X_test  = test24[predictors].copy()
X_2025  = df25[predictors].copy()

for col in numeric_features:
    X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
    X_val[col]   = pd.to_numeric(X_val[col], errors="coerce")
    X_test[col]  = pd.to_numeric(X_test[col], errors="coerce")
    X_2025[col]  = pd.to_numeric(X_2025[col], errors="coerce")


# ============================================================
# 8. Build exact original pipeline
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
            RobustScaler()
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

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ]
)

print("\nTraining model...")
model.fit(X_train, y_train)
print("Training complete.")


# ============================================================
# 9. Probabilities
# ============================================================

val_prob = model.predict_proba(X_val)[:, 1]
test_prob = model.predict_proba(X_test)[:, 1]
forward_prob = model.predict_proba(X_2025)[:, 1]


# ============================================================
# 10. Threshold search on 2024 validation
# ============================================================

threshold_rows = []

for threshold in np.arange(0.01, 1.00, 0.01):

    pred = (val_prob >= threshold).astype(int)

    threshold_rows.append({
        "threshold": round(float(threshold), 2),
        "precision": precision_score(
            y_val, pred, zero_division=0
        ),
        "recall": recall_score(
            y_val, pred, zero_division=0
        ),
        "f1": f1_score(
            y_val, pred, zero_division=0
        )
    })

threshold_df = pd.DataFrame(threshold_rows)


# ============================================================
# 11. F1-optimal threshold
# ============================================================

best_f1_idx = threshold_df["f1"].idxmax()

f1_threshold = float(
    threshold_df.loc[
        best_f1_idx,
        "threshold"
    ]
)

print("\nF1-optimal threshold:")
print(
    threshold_df.loc[
        best_f1_idx
    ]
)


# ============================================================
# 12. Higher-sensitivity thresholds
#
# Choose the HIGHEST threshold that still achieves the
# requested minimum recall on VALIDATION.
#
# Higher threshold = fewer false positives while preserving
# target sensitivity.
# ============================================================

def threshold_for_target_recall(
    threshold_table,
    target_recall
):

    eligible = threshold_table[
        threshold_table["recall"] >= target_recall
    ].copy()

    if len(eligible) == 0:
        return np.nan

    return float(
        eligible["threshold"].max()
    )


recall_targets = [
    0.85,
    0.90,
    0.95
]

sensitivity_thresholds = {}

for target in recall_targets:

    t = threshold_for_target_recall(
        threshold_df,
        target
    )

    sensitivity_thresholds[
        target
    ] = t

print("\nHigher-sensitivity operating points:")

for target, threshold in sensitivity_thresholds.items():

    row = threshold_df[
        threshold_df["threshold"] == threshold
    ].iloc[0]

    print(
        f"Target recall >= {target:.0%}: "
        f"threshold={threshold:.2f}, "
        f"validation recall={row['recall']:.4f}, "
        f"precision={row['precision']:.4f}, "
        f"F1={row['f1']:.4f}"
    )


# ============================================================
# 13. Binary evaluation function
# ============================================================

def evaluate_binary(
    dataset_name,
    y_true,
    prob,
    threshold,
    operating_point
):

    pred = (
        prob >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        pred,
        labels=[0, 1]
    ).ravel()

    return {

        "operating_point":
            operating_point,

        "dataset":
            dataset_name,

        "threshold":
            threshold,

        "n":
            len(y_true),

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
                prob
            ),

        "pr_auc":
            average_precision_score(
                y_true,
                prob
            ),

        "brier":
            brier_score_loss(
                y_true,
                prob
            ),

        "tn":
            int(tn),

        "fp":
            int(fp),

        "fn":
            int(fn),

        "tp":
            int(tp),

        "fpr":
            fp / (fp + tn),

        "fnr":
            fn / (fn + tp)
    }


# ============================================================
# 14. Compare operating points
# ============================================================

binary_results = []

operating_points = {
    "F1-optimal":
        f1_threshold,

    "Recall >= 85%":
        sensitivity_thresholds[0.85],

    "Recall >= 90%":
        sensitivity_thresholds[0.90],

    "Recall >= 95%":
        sensitivity_thresholds[0.95]
}

for label, threshold in operating_points.items():

    if np.isnan(threshold):
        continue

    binary_results.append(
        evaluate_binary(
            "2024 internal test",
            y_test,
            test_prob,
            threshold,
            label
        )
    )

    binary_results.append(
        evaluate_binary(
            "2025 forward temporal",
            y_2025,
            forward_prob,
            threshold,
            label
        )
    )

binary_results_df = pd.DataFrame(
    binary_results
)

print("\n")
print("=" * 90)
print("OPERATING-POINT COMPARISON")
print("=" * 90)

print(
    binary_results_df[
        [
            "operating_point",
            "dataset",
            "threshold",
            "precision",
            "recall",
            "f1",
            "fpr",
            "fnr"
        ]
    ]
    .round(4)
    .to_string(index=False)
)


# ============================================================
# 15. False-negative severity analysis
#
# How far below the EMS-RRI cutoff are missed positives?
# ============================================================

def false_negative_severity(
    dataset_name,
    df,
    y_true,
    prob,
    threshold,
    operating_point
):

    pred = (
        prob >= threshold
    ).astype(int)

    rri = pd.to_numeric(
        df["ems_rri_basic"],
        errors="coerce"
    ).to_numpy()

    y_arr = np.asarray(y_true)

    fn_mask = (
        (y_arr == 1) &
        (pred == 0)
    )

    fn_scores = rri[
        fn_mask
    ]

    if len(fn_scores) == 0:
        return None

    distance_from_cutoff = (
        rri_cutoff - fn_scores
    )

    return {

        "operating_point":
            operating_point,

        "dataset":
            dataset_name,

        "threshold":
            threshold,

        "false_negatives":
            len(fn_scores),

        "median_fn_rri":
            float(
                np.nanmedian(
                    fn_scores
                )
            ),

        "mean_fn_rri":
            float(
                np.nanmean(
                    fn_scores
                )
            ),

        "within_2_points_pct":
            float(
                np.mean(
                    distance_from_cutoff
                    <= 2
                ) * 100
            ),

        "within_5_points_pct":
            float(
                np.mean(
                    distance_from_cutoff
                    <= 5
                ) * 100
            ),

        "below_70_pct":
            float(
                np.mean(
                    fn_scores < 70
                ) * 100
            ),

        "below_60_pct":
            float(
                np.mean(
                    fn_scores < 60
                ) * 100
            )
    }


severity_results = []

for label, threshold in operating_points.items():

    if np.isnan(threshold):
        continue

    r1 = false_negative_severity(
        "2024 internal test",
        test24,
        y_test,
        test_prob,
        threshold,
        label
    )

    r2 = false_negative_severity(
        "2025 forward temporal",
        df25,
        y_2025,
        forward_prob,
        threshold,
        label
    )

    if r1 is not None:
        severity_results.append(r1)

    if r2 is not None:
        severity_results.append(r2)

severity_df = pd.DataFrame(
    severity_results
)

print("\n")
print("=" * 90)
print("FALSE-NEGATIVE SEVERITY")
print("=" * 90)

print(
    severity_df.round(4).to_string(
        index=False
    )
)


# ============================================================
# 16. Three-tier classification
#
# Logic:
#
# Reliable:
#   probability < LOW threshold
#
# Borderline / uncertain:
#   LOW threshold <= probability < HIGH threshold
#
# Low reliability:
#   probability >= HIGH threshold
#
# LOW threshold:
#   selected to achieve >=95% recall on validation.
#
# HIGH threshold:
#   selected to achieve >=90% precision on validation.
#
# This creates a conservative "safe reliable" region and a
# high-confidence "low reliability" region, leaving ambiguous
# cases for selective review.
# ============================================================

three_tier_low_threshold = (
    sensitivity_thresholds[0.95]
)

precision_target = 0.90

eligible_precision = threshold_df[
    threshold_df["precision"] >= precision_target
].copy()

if len(eligible_precision) > 0:

    three_tier_high_threshold = float(
        eligible_precision[
            "threshold"
        ].min()
    )

else:

    three_tier_high_threshold = np.nan


print("\nThree-tier thresholds:")

print(
    "Reliable / borderline boundary:",
    three_tier_low_threshold
)

print(
    "Borderline / low-reliability boundary:",
    three_tier_high_threshold
)


# ============================================================
# 17. Three-tier evaluation function
# ============================================================

def evaluate_three_tier(
    dataset_name,
    y_true,
    prob,
    low_threshold,
    high_threshold
):

    y_arr = np.asarray(y_true)

    tier = np.full(
        len(prob),
        "Borderline",
        dtype=object
    )

    tier[
        prob < low_threshold
    ] = "Reliable"

    tier[
        prob >= high_threshold
    ] = "Low Reliability"


    # --------------------------------------------------------
    # Coverage
    # --------------------------------------------------------

    n_total = len(tier)

    reliable_mask = (
        tier == "Reliable"
    )

    borderline_mask = (
        tier == "Borderline"
    )

    lowrel_mask = (
        tier == "Low Reliability"
    )


    # --------------------------------------------------------
    # Error within reliable region
    #
    # These are true low-reliability records incorrectly
    # allowed into the "Reliable" zone.
    # --------------------------------------------------------

    reliable_positive_rate = (
        y_arr[reliable_mask].mean()
        if reliable_mask.sum() > 0
        else np.nan
    )


    # --------------------------------------------------------
    # Positive predictive value in low-reliability zone
    # --------------------------------------------------------

    lowrel_ppv = (
        y_arr[lowrel_mask].mean()
        if lowrel_mask.sum() > 0
        else np.nan
    )


    # --------------------------------------------------------
    # Fraction of all true low-reliability records captured
    # either as Borderline or Low Reliability
    # --------------------------------------------------------

    positive_mask = (
        y_arr == 1
    )

    captured_for_review = (
        borderline_mask |
        lowrel_mask
    )

    sensitivity_review = (
        captured_for_review[
            positive_mask
        ].mean()
    )


    return {

        "dataset":
            dataset_name,

        "n":
            n_total,

        "reliable_n":
            int(
                reliable_mask.sum()
            ),

        "reliable_pct":
            float(
                reliable_mask.mean() * 100
            ),

        "borderline_n":
            int(
                borderline_mask.sum()
            ),

        "borderline_pct":
            float(
                borderline_mask.mean() * 100
            ),

        "low_reliability_n":
            int(
                lowrel_mask.sum()
            ),

        "low_reliability_pct":
            float(
                lowrel_mask.mean() * 100
            ),

        "low_reliability_rate_inside_reliable_zone":
            float(
                reliable_positive_rate
            ),

        "precision_low_reliability_zone":
            float(
                lowrel_ppv
            ),

        "sensitivity_if_borderline_or_lowrel_reviewed":
            float(
                sensitivity_review
            )
    }


# ============================================================
# 18. Run three-tier analysis
# ============================================================

three_tier_results = []

if not np.isnan(
    three_tier_high_threshold
):

    three_tier_results.append(
        evaluate_three_tier(
            "2024 internal test",
            y_test,
            test_prob,
            three_tier_low_threshold,
            three_tier_high_threshold
        )
    )

    three_tier_results.append(
        evaluate_three_tier(
            "2025 forward temporal",
            y_2025,
            forward_prob,
            three_tier_low_threshold,
            three_tier_high_threshold
        )
    )

three_tier_df = pd.DataFrame(
    three_tier_results
)

print("\n")
print("=" * 90)
print("THREE-TIER SCREENING RESULTS")
print("=" * 90)

if len(three_tier_df) > 0:

    print(
        three_tier_df.round(4).to_string(
            index=False
        )
    )

else:

    print(
        "Could not define high-confidence "
        "low-reliability threshold at "
        "90% precision."
    )


# ============================================================
# 19. Save all outputs
# ============================================================

threshold_df.to_csv(
    f"{OUT}/validation_threshold_search.csv",
    index=False
)

binary_results_df.to_csv(
    f"{OUT}/operating_point_comparison.csv",
    index=False
)

severity_df.to_csv(
    f"{OUT}/false_negative_severity.csv",
    index=False
)

three_tier_df.to_csv(
    f"{OUT}/three_tier_results.csv",
    index=False
)


# ============================================================
# 20. Save configuration
# ============================================================

configuration = {

    "analysis":
        "Boundary and three-tier sensitivity",

    "development_year":
        2024,

    "forward_validation_year":
        2025,

    "ems_rri_cutoff":
        rri_cutoff,

    "f1_optimal_threshold":
        f1_threshold,

    "target_recall_thresholds":
        {
            str(k): v
            for k, v
            in sensitivity_thresholds.items()
        },

    "three_tier": {
        "reliable_upper_probability":
            three_tier_low_threshold,

        "low_reliability_lower_probability":
            three_tier_high_threshold,

        "reliable_threshold_basis":
            "Validation sensitivity >=95%",

        "low_reliability_threshold_basis":
            "Validation precision >=90%"
    },

    "random_state":
        42
}


with open(
    f"{OUT}/analysis_configuration.json",
    "w"
) as f:

    json.dump(
        configuration,
        f,
        indent=2
    )


print("\nSaved outputs to:")
print(OUT)