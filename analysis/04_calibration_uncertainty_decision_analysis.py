# 04 calibration uncertainty decision analysis
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
# EMS-RRI REVISION ANALYSIS 04
# Calibration, Bootstrap Confidence Intervals,
# and Decision Curve Analysis
#
# Development:
#   2024 NEMSIS
#   70% train / 15% validation / 15% internal test
#
# Forward temporal validation:
#   Full 2025 NEMSIS cohort
#
# Outputs:
#   - Main performance metrics
#   - 95% bootstrap confidence intervals
#   - Brier score
#   - Calibration intercept and slope
#   - Expected Calibration Error (ECE)
#   - Calibration tables
#   - Decision Curve Analysis (DCA)
# ============================================================


# ============================================================
# 1. Imports
# ============================================================

import os
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    brier_score_loss,
    log_loss
)

from xgboost import XGBClassifier

from scipy.special import expit, logit
from scipy.optimize import minimize

warnings.filterwarnings("ignore")


# ============================================================
# 2. Paths
# ============================================================

BASE = str(ANALYSIS_ROOT)

DATA_2024 = f"{BASE}/datasets/nemsis_2024.parquet"
DATA_2025 = f"{BASE}/datasets/nemsis_2025.parquet"

OUT = (
    f"{BASE}/revision_analysis/outputs/"
    "04_calibration_confidence_intervals"
)

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

print("\nPredictor specification:")
print("Numeric:", len(numeric_features))
print("Categorical:", len(categorical_features))
print("Total:", len(predictors))


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

print("\n2024 development split:")
print(f"Train:         {len(train24):,}")
print(f"Validation:    {len(val24):,}")
print(f"Internal test: {len(test24):,}")
print(f"2025 forward:  {len(df25):,}")


# ============================================================
# 6. EMS-RRI cutoff derived from 2024 TRAIN ONLY
# ============================================================

rri_cutoff = float(
    train24["ems_rri_basic"].quantile(0.20)
)

print("\nEMS-RRI cutoff:")
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

    X_train[col] = pd.to_numeric(
        X_train[col],
        errors="coerce"
    )

    X_val[col] = pd.to_numeric(
        X_val[col],
        errors="coerce"
    )

    X_test[col] = pd.to_numeric(
        X_test[col],
        errors="coerce"
    )

    X_2025[col] = pd.to_numeric(
        X_2025[col],
        errors="coerce"
    )


# ============================================================
# 8. Exact preprocessing pipeline
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


# ============================================================
# 9. Exact XGBoost model
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
# 10. Train
# ============================================================

print("\nTraining model...")

model.fit(
    X_train,
    y_train
)

print("Training complete.")


# ============================================================
# 11. Prediction probabilities
# ============================================================

val_prob = model.predict_proba(X_val)[:, 1]
test_prob = model.predict_proba(X_test)[:, 1]
forward_prob = model.predict_proba(X_2025)[:, 1]


# ============================================================
# 12. Confirm validation-selected F1 threshold
# ============================================================

threshold_rows = []

for threshold in np.arange(
    0.01,
    1.00,
    0.01
):

    pred = (
        val_prob >= threshold
    ).astype(int)

    threshold_rows.append({
        "threshold":
            round(float(threshold), 2),

        "f1":
            f1_score(
                y_val,
                pred,
                zero_division=0
            )
    })

threshold_df = pd.DataFrame(
    threshold_rows
)

best_idx = threshold_df[
    "f1"
].idxmax()

best_threshold = float(
    threshold_df.loc[
        best_idx,
        "threshold"
    ]
)

print("\nSelected validation threshold:")
print(best_threshold)

assert abs(
    best_threshold - 0.37
) < 1e-9


# ============================================================
# 13. Main evaluation function
# ============================================================

def evaluate_metrics(
    y_true,
    prob,
    threshold
):

    y_true = np.asarray(
        y_true
    )

    pred = (
        prob >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        pred,
        labels=[0, 1]
    ).ravel()

    return {

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

        "log_loss":
            log_loss(
                y_true,
                prob
            ),

        "fpr":
            fp / (fp + tn),

        "fnr":
            fn / (fn + tp)
    }


internal_metrics = evaluate_metrics(
    y_test,
    test_prob,
    best_threshold
)

forward_metrics = evaluate_metrics(
    y_2025,
    forward_prob,
    best_threshold
)

print("\n2024 internal-test metrics:")
print(
    pd.Series(
        internal_metrics
    ).round(6)
)

print("\n2025 forward-validation metrics:")
print(
    pd.Series(
        forward_metrics
    ).round(6)
)


# ============================================================
# 14. Calibration intercept and slope
#
# Ideal:
#   intercept = 0
#   slope = 1
#
# Fit:
# logit(observed outcome) =
#     intercept + slope * logit(predicted probability)
# ============================================================

def calibration_intercept_slope(
    y_true,
    prob
):

    y = np.asarray(
        y_true,
        dtype=float
    )

    p = np.clip(
        np.asarray(
            prob,
            dtype=float
        ),
        1e-6,
        1 - 1e-6
    )

    x = logit(p)

    def negative_log_likelihood(
        beta
    ):

        linear = (
            beta[0] +
            beta[1] * x
        )

        fitted = expit(
            linear
        )

        fitted = np.clip(
            fitted,
            1e-10,
            1 - 1e-10
        )

        return -np.sum(
            y * np.log(fitted) +
            (1 - y) *
            np.log(1 - fitted)
        )

    result = minimize(
        negative_log_likelihood,
        x0=np.array([
            0.0,
            1.0
        ]),
        method="BFGS"
    )

    return (
        float(
            result.x[0]
        ),
        float(
            result.x[1]
        )
    )


internal_cal_intercept, internal_cal_slope = (
    calibration_intercept_slope(
        y_test,
        test_prob
    )
)

forward_cal_intercept, forward_cal_slope = (
    calibration_intercept_slope(
        y_2025,
        forward_prob
    )
)

print("\nCalibration:")
print(
    "2024 internal test - "
    f"intercept={internal_cal_intercept:.4f}, "
    f"slope={internal_cal_slope:.4f}"
)

print(
    "2025 forward - "
    f"intercept={forward_cal_intercept:.4f}, "
    f"slope={forward_cal_slope:.4f}"
)


# ============================================================
# 15. Calibration table + ECE
# ============================================================

def calibration_table(
    y_true,
    prob,
    n_bins=10
):

    df = pd.DataFrame({
        "y":
            np.asarray(
                y_true
            ),

        "prob":
            np.asarray(
                prob
            )
    })

    # Quantile bins keep reasonable sample
    # sizes across probability ranges
    df["bin"] = pd.qcut(
        df["prob"],
        q=n_bins,
        duplicates="drop"
    )

    table = (
        df
        .groupby(
            "bin",
            observed=True
        )
        .agg(
            n=("y", "size"),
            mean_predicted=(
                "prob",
                "mean"
            ),
            observed_rate=(
                "y",
                "mean"
            )
        )
        .reset_index()
    )

    table[
        "absolute_error"
    ] = np.abs(
        table[
            "mean_predicted"
        ] -
        table[
            "observed_rate"
        ]
    )

    total = table["n"].sum()

    ece = np.sum(
        (
            table["n"] /
            total
        ) *
        table[
            "absolute_error"
        ]
    )

    return (
        table,
        float(ece)
    )


internal_cal_table, internal_ece = (
    calibration_table(
        y_test,
        test_prob,
        n_bins=10
    )
)

forward_cal_table, forward_ece = (
    calibration_table(
        y_2025,
        forward_prob,
        n_bins=10
    )
)

print("\nExpected Calibration Error:")
print(
    f"2024 internal test: {internal_ece:.6f}"
)

print(
    f"2025 forward:      {forward_ece:.6f}"
)


# ============================================================
# 16. Bootstrap confidence intervals
#
# Predictions are frozen.
# We bootstrap record-level pairs:
#   (outcome, predicted probability)
#
# Internal test:
#   500 bootstrap replicates
#
# Forward cohort:
#   200 bootstrap replicates
#
# Increase later only if desired.
# ============================================================

BOOTSTRAP_SEED = 42


def bootstrap_confidence_intervals(
    y_true,
    prob,
    threshold,
    n_bootstrap,
    seed=42
):

    y = np.asarray(
        y_true
    )

    p = np.asarray(
        prob
    )

    n = len(y)

    rng = np.random.default_rng(
        seed
    )

    rows = []

    for b in range(
        n_bootstrap
    ):

        idx = rng.integers(
            0,
            n,
            size=n
        )

        y_b = y[idx]
        p_b = p[idx]

        # Rare safety check
        if len(
            np.unique(
                y_b
            )
        ) < 2:
            continue

        metrics = evaluate_metrics(
            y_b,
            p_b,
            threshold
        )

        metrics[
            "bootstrap_iteration"
        ] = b + 1

        rows.append(
            metrics
        )

        if (
            (b + 1) % 50 == 0
            or b == 0
        ):
            print(
                f"Bootstrap "
                f"{b+1}/{n_bootstrap}"
            )

    boot_df = pd.DataFrame(
        rows
    )

    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "brier",
        "fpr",
        "fnr"
    ]

    ci_rows = []

    for metric in metric_names:

        values = boot_df[
            metric
        ].dropna()

        ci_rows.append({

            "metric":
                metric,

            "bootstrap_mean":
                values.mean(),

            "lower_95":
                values.quantile(
                    0.025
                ),

            "upper_95":
                values.quantile(
                    0.975
                )
        })

    return (
        boot_df,
        pd.DataFrame(
            ci_rows
        )
    )


print("\n")
print("=" * 80)
print("BOOTSTRAP: 2024 INTERNAL TEST")
print("=" * 80)

internal_boot, internal_ci = (
    bootstrap_confidence_intervals(
        y_test,
        test_prob,
        best_threshold,
        n_bootstrap=500,
        seed=42
    )
)


print("\n")
print("=" * 80)
print("BOOTSTRAP: 2025 FORWARD VALIDATION")
print("=" * 80)

forward_boot, forward_ci = (
    bootstrap_confidence_intervals(
        y_2025,
        forward_prob,
        best_threshold,
        n_bootstrap=200,
        seed=43
    )
)


# ============================================================
# 17. Attach point estimates to CI tables
# ============================================================

internal_point = pd.Series(
    internal_metrics
)

forward_point = pd.Series(
    forward_metrics
)


def attach_point_estimates(
    ci_df,
    point_series
):

    df = ci_df.copy()

    df[
        "point_estimate"
    ] = df[
        "metric"
    ].map(
        point_series
    )

    return df[
        [
            "metric",
            "point_estimate",
            "lower_95",
            "upper_95",
            "bootstrap_mean"
        ]
    ]


internal_ci_final = (
    attach_point_estimates(
        internal_ci,
        internal_point
    )
)

forward_ci_final = (
    attach_point_estimates(
        forward_ci,
        forward_point
    )
)

print("\n")
print("=" * 80)
print("2024 INTERNAL TEST: 95% CONFIDENCE INTERVALS")
print("=" * 80)

print(
    internal_ci_final
    .round(6)
    .to_string(
        index=False
    )
)

print("\n")
print("=" * 80)
print("2025 FORWARD VALIDATION: 95% CONFIDENCE INTERVALS")
print("=" * 80)

print(
    forward_ci_final
    .round(6)
    .to_string(
        index=False
    )
)


# ============================================================
# 18. Calibration summary
# ============================================================

calibration_summary = pd.DataFrame([
    {
        "dataset":
            "2024 internal test",

        "n":
            len(y_test),

        "prevalence":
            np.mean(y_test),

        "brier":
            internal_metrics[
                "brier"
            ],

        "log_loss":
            internal_metrics[
                "log_loss"
            ],

        "calibration_intercept":
            internal_cal_intercept,

        "calibration_slope":
            internal_cal_slope,

        "ece":
            internal_ece
    },

    {
        "dataset":
            "2025 forward temporal",

        "n":
            len(y_2025),

        "prevalence":
            np.mean(y_2025),

        "brier":
            forward_metrics[
                "brier"
            ],

        "log_loss":
            forward_metrics[
                "log_loss"
            ],

        "calibration_intercept":
            forward_cal_intercept,

        "calibration_slope":
            forward_cal_slope,

        "ece":
            forward_ece
    }
])

print("\n")
print("=" * 80)
print("CALIBRATION SUMMARY")
print("=" * 80)

print(
    calibration_summary
    .round(6)
    .to_string(
        index=False
    )
)


# ============================================================
# 19. Decision Curve Analysis
#
# Net benefit:
#
#   TP/N - FP/N * pt/(1-pt)
#
# Compare:
#   model
#   treat all
#   treat none
#
# Here "treat" means route the record to
# reliability review / intervention.
# ============================================================

def decision_curve(
    y_true,
    prob,
    thresholds
):

    y = np.asarray(
        y_true
    )

    p = np.asarray(
        prob
    )

    n = len(y)

    prevalence = y.mean()

    rows = []

    for pt in thresholds:

        pred = (
            p >= pt
        ).astype(int)

        tn, fp, fn, tp = (
            confusion_matrix(
                y,
                pred,
                labels=[0, 1]
            ).ravel()
        )

        odds = (
            pt /
            (1 - pt)
        )

        model_nb = (
            tp / n
            -
            (fp / n) * odds
        )

        treat_all_nb = (
            prevalence
            -
            (1 - prevalence)
            * odds
        )

        rows.append({
            "threshold_probability":
                pt,

            "model_net_benefit":
                model_nb,

            "review_all_net_benefit":
                treat_all_nb,

            "review_none_net_benefit":
                0.0
        })

    return pd.DataFrame(
        rows
    )


dca_thresholds = np.arange(
    0.05,
    0.81,
    0.01
)

internal_dca = decision_curve(
    y_test,
    test_prob,
    dca_thresholds
)

forward_dca = decision_curve(
    y_2025,
    forward_prob,
    dca_thresholds
)


# ============================================================
# 20. DCA summaries at clinically relevant model thresholds
# ============================================================

def nearest_dca_rows(
    dca_df,
    requested_thresholds
):

    rows = []

    for t in requested_thresholds:

        idx = (
            dca_df[
                "threshold_probability"
            ] - t
        ).abs().idxmin()

        row = dca_df.loc[
            idx
        ].copy()

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


selected_dca_thresholds = [
    0.14,
    0.22,
    0.29,
    0.37,
    0.63
]

internal_dca_selected = (
    nearest_dca_rows(
        internal_dca,
        selected_dca_thresholds
    )
)

forward_dca_selected = (
    nearest_dca_rows(
        forward_dca,
        selected_dca_thresholds
    )
)

print("\n")
print("=" * 80)
print("DCA — 2025 FORWARD VALIDATION")
print("=" * 80)

print(
    forward_dca_selected
    .round(6)
    .to_string(
        index=False
    )
)


# ============================================================
# 21. Calibration plots
# ============================================================

plt.figure(figsize=(6, 6))

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect calibration"
)

plt.plot(
    internal_cal_table[
        "mean_predicted"
    ],
    internal_cal_table[
        "observed_rate"
    ],
    marker="o",
    label="2024 internal test"
)

plt.plot(
    forward_cal_table[
        "mean_predicted"
    ],
    forward_cal_table[
        "observed_rate"
    ],
    marker="o",
    label="2025 forward validation"
)

plt.xlabel(
    "Mean predicted probability"
)

plt.ylabel(
    "Observed low-reliability rate"
)

plt.title(
    "Calibration of Low-Reliability Prediction"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    f"{OUT}/calibration_plot.png",
    dpi=300
)

plt.show()


# ============================================================
# 22. DCA plot — forward validation
# ============================================================

plt.figure(figsize=(7, 5))

plt.plot(
    forward_dca[
        "threshold_probability"
    ],
    forward_dca[
        "model_net_benefit"
    ],
    label="EMS-RRI prediction model"
)

plt.plot(
    forward_dca[
        "threshold_probability"
    ],
    forward_dca[
        "review_all_net_benefit"
    ],
    linestyle="--",
    label="Review all"
)

plt.plot(
    forward_dca[
        "threshold_probability"
    ],
    forward_dca[
        "review_none_net_benefit"
    ],
    linestyle=":",
    label="Review none"
)

plt.xlabel(
    "Threshold probability"
)

plt.ylabel(
    "Net benefit"
)

plt.title(
    "Decision Curve Analysis — 2025 Forward Validation"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    f"{OUT}/dca_forward_validation.png",
    dpi=300
)

plt.show()


# ============================================================
# 23. Save outputs
# ============================================================

pd.DataFrame(
    [
        {
            "dataset":
                "2024 internal test",
            **internal_metrics
        },

        {
            "dataset":
                "2025 forward temporal",
            **forward_metrics
        }
    ]
).to_csv(
    f"{OUT}/main_metrics.csv",
    index=False
)

internal_ci_final.to_csv(
    f"{OUT}/internal_test_confidence_intervals.csv",
    index=False
)

forward_ci_final.to_csv(
    f"{OUT}/forward_validation_confidence_intervals.csv",
    index=False
)

internal_boot.to_csv(
    f"{OUT}/internal_bootstrap_replicates.csv",
    index=False
)

forward_boot.to_csv(
    f"{OUT}/forward_bootstrap_replicates.csv",
    index=False
)

calibration_summary.to_csv(
    f"{OUT}/calibration_summary.csv",
    index=False
)

internal_cal_table.to_csv(
    f"{OUT}/internal_calibration_bins.csv",
    index=False
)

forward_cal_table.to_csv(
    f"{OUT}/forward_calibration_bins.csv",
    index=False
)

internal_dca.to_csv(
    f"{OUT}/internal_dca.csv",
    index=False
)

forward_dca.to_csv(
    f"{OUT}/forward_dca.csv",
    index=False
)

internal_dca_selected.to_csv(
    f"{OUT}/internal_dca_selected_thresholds.csv",
    index=False
)

forward_dca_selected.to_csv(
    f"{OUT}/forward_dca_selected_thresholds.csv",
    index=False
)


# ============================================================
# 24. Save configuration
# ============================================================

configuration = {

    "analysis":
        "Calibration, confidence intervals, and DCA",

    "development_year":
        2024,

    "forward_validation_year":
        2025,

    "ems_rri_cutoff":
        rri_cutoff,

    "classification_threshold":
        best_threshold,

    "bootstrap_internal_replicates":
        500,

    "bootstrap_forward_replicates":
        200,

    "calibration_bins":
        10,

    "DCA_threshold_range":
        "0.05 to 0.80",

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