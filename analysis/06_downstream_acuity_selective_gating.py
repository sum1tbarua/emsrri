# 06 downstream acuity selective gating
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
# EMS-RRI REVISION ANALYSIS 08
# Forward Downstream High-Acuity Validation
# + Selective-Prediction / Abstention Analysis
#
# PRIMARY DOWNSTREAM TARGET:
#   Positive:
#       4219001 = Critical
#       4219003 = Emergent
#
#   Negative:
#       4219005 = Lower Acuity
#
# "Dead without Resuscitation Efforts" (4219007) remains
# handled separately in Analysis 07 sensitivity analysis.
#
# TEMPORAL DESIGN:
#   2024 = development
#       70% train
#       15% validation
#       15% internal test
#
#   2025 = full forward temporal validation
#
# PART A:
#   Baseline vs Baseline + EMS-RRI
#
# PART B:
#   EMS-RRI used ONLY as an abstention gate for the
#   baseline downstream model.
#
# IMPORTANT:
# Selective-gating results demonstrate retrospective
# information utility, not prospective operational deployment.
# ============================================================


# ============================================================
# 1. Imports
# ============================================================

from pathlib import Path
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
    brier_score_loss,
    confusion_matrix
)

from xgboost import XGBClassifier


# ============================================================
# 2. Paths
# ============================================================

BASE = Path(
    str(ANALYSIS_ROOT)
)

DATA_2024 = (
    BASE /
    "datasets/nemsis_2024.parquet"
)

DATA_2025 = (
    BASE /
    "datasets/nemsis_2025.parquet"
)

OUT = (
    BASE /
    "revision_analysis/outputs/"
    "08_forward_downstream_selective_gate"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

print("Output directory:")
print(OUT)


# ============================================================
# 3. Load full datasets
# ============================================================

df24 = pd.read_parquet(
    DATA_2024
)

df25 = pd.read_parquet(
    DATA_2025
)

print("\nFull cohort sizes:")
print("2024:", len(df24))
print("2025:", len(df25))


# ============================================================
# 4. Reconstruct ORIGINAL downstream high-acuity target
# ============================================================

HIGH_CODES = {
    "4219001",  # Critical
    "4219003",  # Emergent
}

LOW_CODES = {
    "4219005",  # Lower Acuity
}


def make_high_acuity_target(df):

    values = (
        df["eDisposition_19"]
        .astype(str)
        .str.strip()
    )

    y = pd.Series(
        np.nan,
        index=df.index,
        dtype="float64",
        name="high_final_acuity"
    )

    y.loc[
        values.isin(
            HIGH_CODES
        )
    ] = 1

    y.loc[
        values.isin(
            LOW_CODES
        )
    ] = 0

    return y


df24[
    "high_final_acuity"
] = make_high_acuity_target(
    df24
)

df25[
    "high_final_acuity"
] = make_high_acuity_target(
    df25
)


# ============================================================
# 5. Restrict to eligible target records
# ============================================================

acuity24 = (
    df24.loc[
        df24[
            "high_final_acuity"
        ].notna()
    ]
    .copy()
)

acuity25 = (
    df25.loc[
        df25[
            "high_final_acuity"
        ].notna()
    ]
    .copy()
)

acuity24[
    "high_final_acuity"
] = (
    acuity24[
        "high_final_acuity"
    ]
    .astype("int8")
)

acuity25[
    "high_final_acuity"
] = (
    acuity25[
        "high_final_acuity"
    ]
    .astype("int8")
)

print("\nEligible downstream cohorts:")
print(
    "2024:",
    len(acuity24),
    "prevalence:",
    acuity24[
        "high_final_acuity"
    ].mean()
)

print(
    "2025:",
    len(acuity25),
    "prevalence:",
    acuity25[
        "high_final_acuity"
    ].mean()
)


# ============================================================
# 6. Create FORWARD development split from 2024
#
# Stratification is used because this downstream target is
# substantially imbalanced.
# ============================================================

train24, temp24 = train_test_split(
    acuity24,
    test_size=0.30,
    random_state=42,
    shuffle=True,
    stratify=acuity24[
        "high_final_acuity"
    ]
)

val24, test24 = train_test_split(
    temp24,
    test_size=0.50,
    random_state=42,
    shuffle=True,
    stratify=temp24[
        "high_final_acuity"
    ]
)

print("\nForward downstream split:")
print(
    f"2024 train:         {len(train24):,}"
)

print(
    f"2024 validation:    {len(val24):,}"
)

print(
    f"2024 internal test: {len(test24):,}"
)

print(
    f"2025 forward:       {len(acuity25):,}"
)


for name, d in [
    ("2024 train", train24),
    ("2024 validation", val24),
    ("2024 internal", test24),
    ("2025 forward", acuity25)
]:

    print(
        name,
        "prevalence =",
        round(
            d[
                "high_final_acuity"
            ].mean(),
            6
        )
    )


# ============================================================
# 7. Original downstream predictor specification
# ============================================================

baseline_predictors = [

    # Early operational timing
    "computed_EMSChuteTimeMin",
    "computed_EMSDispatchCenterTimeSec",
    "computed_EMSSceneResponseTimeMin",
    "computed_EMSSceneToPatientTimeMin",
    "computed_EMSSystemResponseTimeMin",

    # Geographic context
    "computed_NasemsoRegion",
    "computed_USCensusDivision",
    "computed_USCensusRegion",
    "computed_Urbanicity",

    # Patient
    "computed_ageinyear",

    # Clinical context
    "eArrest_05",
    "eSituation_01",
    "eSituation_02",
    "eSituation_07",
    "eSituation_08",
    "eSituation_13",
    "eSituation_18",
    "eSituation_20",
]

rri_predictors = (
    baseline_predictors +
    ["ems_rri_basic"]
)

assert len(
    baseline_predictors
) == 18

assert len(
    rri_predictors
) == 19


baseline_numeric = [
    "computed_EMSChuteTimeMin",
    "computed_EMSDispatchCenterTimeSec",
    "computed_EMSSceneResponseTimeMin",
    "computed_EMSSceneToPatientTimeMin",
    "computed_EMSSystemResponseTimeMin",
    "computed_ageinyear",
]

baseline_categorical = [
    c
    for c in baseline_predictors
    if c not in baseline_numeric
]

rri_numeric = (
    baseline_numeric +
    ["ems_rri_basic"]
)

rri_categorical = (
    baseline_categorical.copy()
)


# ============================================================
# 8. Model builder
# ============================================================

def build_model(
    numeric_features,
    categorical_features
):

    num_pipeline = Pipeline(
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

    cat_pipeline = Pipeline(
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
                num_pipeline,
                numeric_features
            ),
            (
                "categorical",
                cat_pipeline,
                categorical_features
            )
        ]
    )

    clf = XGBClassifier(
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
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                clf
            )
        ]
    )


# ============================================================
# 9. Input preparation
# ============================================================

def prepare_X(
    df,
    predictors,
    numeric_features
):

    X = df[
        predictors
    ].copy()

    for col in numeric_features:

        X[col] = pd.to_numeric(
            X[col],
            errors="coerce"
        )

    return X


def get_y(df):

    return (
        df[
            "high_final_acuity"
        ]
        .astype(int)
        .to_numpy()
    )


y_train = get_y(
    train24
)

y_val = get_y(
    val24
)

y_test = get_y(
    test24
)

y_forward = get_y(
    acuity25
)


X_train_base = prepare_X(
    train24,
    baseline_predictors,
    baseline_numeric
)

X_val_base = prepare_X(
    val24,
    baseline_predictors,
    baseline_numeric
)

X_test_base = prepare_X(
    test24,
    baseline_predictors,
    baseline_numeric
)

X_forward_base = prepare_X(
    acuity25,
    baseline_predictors,
    baseline_numeric
)


X_train_rri = prepare_X(
    train24,
    rri_predictors,
    rri_numeric
)

X_val_rri = prepare_X(
    val24,
    rri_predictors,
    rri_numeric
)

X_test_rri = prepare_X(
    test24,
    rri_predictors,
    rri_numeric
)

X_forward_rri = prepare_X(
    acuity25,
    rri_predictors,
    rri_numeric
)


# ============================================================
# 10. Fit forward downstream models
# ============================================================

baseline_model = build_model(
    baseline_numeric,
    baseline_categorical
)

rri_model = build_model(
    rri_numeric,
    rri_categorical
)

print("\nTraining baseline model...")

baseline_model.fit(
    X_train_base,
    y_train
)

print("Baseline complete.")

print("\nTraining EMS-RRI-enhanced model...")

rri_model.fit(
    X_train_rri,
    y_train
)

print("EMS-RRI-enhanced model complete.")


# ============================================================
# 11. Validation threshold selection
# ============================================================

def find_best_f1_threshold(
    y,
    prob
):

    rows = []

    for t in np.arange(
        0.01,
        1.00,
        0.01
    ):

        pred = (
            prob >= t
        ).astype(int)

        rows.append({

            "threshold":
                round(
                    float(t),
                    2
                ),

            "precision":
                precision_score(
                    y,
                    pred,
                    zero_division=0
                ),

            "recall":
                recall_score(
                    y,
                    pred,
                    zero_division=0
                ),

            "f1":
                f1_score(
                    y,
                    pred,
                    zero_division=0
                )
        })

    table = pd.DataFrame(
        rows
    )

    best = table.loc[
        table[
            "f1"
        ].idxmax()
    ]

    return (
        float(
            best[
                "threshold"
            ]
        ),
        table
    )


base_val_prob = (
    baseline_model
    .predict_proba(
        X_val_base
    )[:, 1]
)

rri_val_prob = (
    rri_model
    .predict_proba(
        X_val_rri
    )[:, 1]
)

base_threshold, base_search = (
    find_best_f1_threshold(
        y_val,
        base_val_prob
    )
)

rri_threshold, rri_search = (
    find_best_f1_threshold(
        y_val,
        rri_val_prob
    )
)

print("\nSelected thresholds:")
print(
    "Baseline:",
    base_threshold
)

print(
    "Baseline + EMS-RRI:",
    rri_threshold
)


# ============================================================
# 12. General metric function
# ============================================================

def evaluate(
    y,
    prob,
    threshold
):

    y = np.asarray(
        y
    )

    prob = np.asarray(
        prob
    )

    pred = (
        prob >= threshold
    ).astype(int)

    tn, fp, fn, tp = (
        confusion_matrix(
            y,
            pred,
            labels=[0, 1]
        ).ravel()
    )

    return {

        "n":
            len(y),

        "prevalence":
            float(
                np.mean(y)
            ),

        "accuracy":
            accuracy_score(
                y,
                pred
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                y,
                pred
            ),

        "precision":
            precision_score(
                y,
                pred,
                zero_division=0
            ),

        "recall":
            recall_score(
                y,
                pred,
                zero_division=0
            ),

        "f1":
            f1_score(
                y,
                pred,
                zero_division=0
            ),

        "roc_auc":
            roc_auc_score(
                y,
                prob
            ),

        "pr_auc":
            average_precision_score(
                y,
                prob
            ),

        "brier":
            brier_score_loss(
                y,
                prob
            ),

        "error_rate":
            float(
                np.mean(
                    pred != y
                )
            ),

        "tn":
            int(tn),

        "fp":
            int(fp),

        "fn":
            int(fn),

        "tp":
            int(tp)
    }


# ============================================================
# 13. Generate probabilities
# ============================================================

base_test_prob = (
    baseline_model
    .predict_proba(
        X_test_base
    )[:, 1]
)

rri_test_prob = (
    rri_model
    .predict_proba(
        X_test_rri
    )[:, 1]
)

base_forward_prob = (
    baseline_model
    .predict_proba(
        X_forward_base
    )[:, 1]
)

rri_forward_prob = (
    rri_model
    .predict_proba(
        X_forward_rri
    )[:, 1]
)


# ============================================================
# 14. Forward additive-feature comparison
# ============================================================

model_rows = []

for dataset_name, y, base_prob, rri_prob in [

    (
        "2024 internal test",
        y_test,
        base_test_prob,
        rri_test_prob
    ),

    (
        "2025 forward temporal",
        y_forward,
        base_forward_prob,
        rri_forward_prob
    )
]:

    base_metrics = evaluate(
        y,
        base_prob,
        base_threshold
    )

    enhanced_metrics = evaluate(
        y,
        rri_prob,
        rri_threshold
    )

    model_rows.append({
        "dataset":
            dataset_name,

        "model":
            "Baseline",

        "threshold":
            base_threshold,

        **base_metrics
    })

    model_rows.append({
        "dataset":
            dataset_name,

        "model":
            "Baseline + EMS-RRI",

        "threshold":
            rri_threshold,

        **enhanced_metrics
    })


model_results = pd.DataFrame(
    model_rows
)

print("\n")
print("=" * 110)
print("FORWARD DOWNSTREAM MODEL RESULTS")
print("=" * 110)

print(
    model_results
    .round(6)
    .to_string(
        index=False
    )
)


# ============================================================
# 15. Incremental forward differences
# ============================================================

metric_names = [
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "brier",
    "error_rate"
]

difference_rows = []

for dataset in model_results[
    "dataset"
].unique():

    subset = (
        model_results[
            model_results[
                "dataset"
            ] == dataset
        ]
        .set_index(
            "model"
        )
    )

    for metric in metric_names:

        difference_rows.append({

            "dataset":
                dataset,

            "metric":
                metric,

            "baseline":
                subset.loc[
                    "Baseline",
                    metric
                ],

            "ems_rri":
                subset.loc[
                    "Baseline + EMS-RRI",
                    metric
                ],

            "difference":
                (
                    subset.loc[
                        "Baseline + EMS-RRI",
                        metric
                    ]
                    -
                    subset.loc[
                        "Baseline",
                        metric
                    ]
                )
        })


difference_df = pd.DataFrame(
    difference_rows
)

print("\n")
print("=" * 110)
print("FORWARD EMS-RRI INCREMENTAL DIFFERENCES")
print("=" * 110)

print(
    difference_df
    .round(6)
    .to_string(
        index=False
    )
)


# ============================================================
# 16. SELECTIVE-PREDICTION GATE
#
# EMS-RRI is NOT supplied as a model feature here.
#
# The baseline downstream model makes predictions.
# EMS-RRI determines whether those predictions are accepted
# or abstained.
#
# Primary frozen EMS-RRI cutoff:
#       low reliability <= 81.8182
#
# This cutoff comes from the primary reliability analysis.
# ============================================================

PRIMARY_RRI_CUTOFF = 81.8182

rri_scores_forward = pd.to_numeric(
    acuity25[
        "ems_rri_basic"
    ],
    errors="coerce"
).to_numpy()

base_forward_pred = (
    base_forward_prob >=
    base_threshold
).astype(int)


# ============================================================
# 17. Gate evaluation helper
# ============================================================

def safe_metrics(
    y,
    prob,
    threshold
):

    if len(y) == 0:
        return None

    pred = (
        prob >= threshold
    ).astype(int)

    result = {

        "n":
            len(y),

        "prevalence":
            float(
                np.mean(y)
            ),

        "accuracy":
            accuracy_score(
                y,
                pred
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                y,
                pred
            ),

        "precision":
            precision_score(
                y,
                pred,
                zero_division=0
            ),

        "recall":
            recall_score(
                y,
                pred,
                zero_division=0
            ),

        "f1":
            f1_score(
                y,
                pred,
                zero_division=0
            ),

        "brier":
            brier_score_loss(
                y,
                prob
            ),

        "error_rate":
            float(
                np.mean(
                    pred != y
                )
            )
    }

    if len(
        np.unique(
            y
        )
    ) == 2:

        result[
            "roc_auc"
        ] = roc_auc_score(
            y,
            prob
        )

        result[
            "pr_auc"
        ] = average_precision_score(
            y,
            prob
        )

    else:

        result[
            "roc_auc"
        ] = np.nan

        result[
            "pr_auc"
        ] = np.nan

    return result


# ============================================================
# 18. Primary gate: cutoff 81.8182
# ============================================================

accepted_mask = (
    rri_scores_forward >
    PRIMARY_RRI_CUTOFF
)

abstained_mask = (
    ~accepted_mask
)

overall_metrics = safe_metrics(
    y_forward,
    base_forward_prob,
    base_threshold
)

accepted_metrics = safe_metrics(
    y_forward[
        accepted_mask
    ],
    base_forward_prob[
        accepted_mask
    ],
    base_threshold
)

abstained_metrics = safe_metrics(
    y_forward[
        abstained_mask
    ],
    base_forward_prob[
        abstained_mask
    ],
    base_threshold
)


primary_gate_summary = pd.DataFrame([
    {
        "group":
            "All records",

        "coverage":
            1.0,

        "abstention_rate":
            0.0,

        **overall_metrics
    },

    {
        "group":
            "Accepted: EMS-RRI > 81.8182",

        "coverage":
            float(
                accepted_mask.mean()
            ),

        "abstention_rate":
            float(
                abstained_mask.mean()
            ),

        **accepted_metrics
    },

    {
        "group":
            "Abstained: EMS-RRI <= 81.8182",

        "coverage":
            float(
                abstained_mask.mean()
            ),

        "abstention_rate":
            np.nan,

        **abstained_metrics
    }
])


print("\n")
print("=" * 110)
print("PRIMARY SELECTIVE-PREDICTION GATE — 2025 FORWARD")
print("=" * 110)

print(
    primary_gate_summary
    .round(6)
    .to_string(
        index=False
    )
)


# ============================================================
# 19. Does the gate concentrate downstream errors?
# ============================================================

error_indicator = (
    base_forward_pred !=
    y_forward
).astype(int)

accepted_error_rate = float(
    error_indicator[
        accepted_mask
    ].mean()
)

abstained_error_rate = float(
    error_indicator[
        abstained_mask
    ].mean()
)

error_risk_ratio = (
    abstained_error_rate /
    accepted_error_rate
)

total_errors = int(
    error_indicator.sum()
)

errors_in_abstained = int(
    error_indicator[
        abstained_mask
    ].sum()
)

fraction_errors_captured = (
    errors_in_abstained /
    total_errors
)


error_concentration = pd.DataFrame([
    {
        "accepted_n":
            int(
                accepted_mask.sum()
            ),

        "abstained_n":
            int(
                abstained_mask.sum()
            ),

        "coverage":
            float(
                accepted_mask.mean()
            ),

        "abstention_rate":
            float(
                abstained_mask.mean()
            ),

        "accepted_error_rate":
            accepted_error_rate,

        "abstained_error_rate":
            abstained_error_rate,

        "abstained_vs_accepted_error_risk_ratio":
            error_risk_ratio,

        "total_downstream_errors":
            total_errors,

        "errors_in_abstained_records":
            errors_in_abstained,

        "fraction_all_errors_captured_by_abstention":
            fraction_errors_captured
    }
])


print("\n")
print("=" * 110)
print("ERROR CONCENTRATION UNDER EMS-RRI GATING")
print("=" * 110)

print(
    error_concentration
    .round(6)
    .to_string(
        index=False
    )
)


# ============================================================
# 20. Gate sensitivity across previously evaluated RRI cutoffs
#
# These are NOT newly optimized on 2025.
# They came from 2024-training percentile sensitivity:
#
# 72.7273 = 10th-percentile-derived definition
# 80.0000 = 15th-percentile-derived definition
# 81.8182 = 20th/25th/30th percentile tie point
# ============================================================

GATE_CUTOFFS = [
    72.7273,
    80.0000,
    81.8182
]

gate_rows = []

for cutoff in GATE_CUTOFFS:

    accept = (
        rri_scores_forward >
        cutoff
    )

    abstain = (
        ~accept
    )

    accepted = safe_metrics(
        y_forward[
            accept
        ],
        base_forward_prob[
            accept
        ],
        base_threshold
    )

    abstained = safe_metrics(
        y_forward[
            abstain
        ],
        base_forward_prob[
            abstain
        ],
        base_threshold
    )

    gate_rows.append({

        "rri_cutoff":
            cutoff,

        "coverage":
            float(
                accept.mean()
            ),

        "abstention_rate":
            float(
                abstain.mean()
            ),

        "accepted_n":
            int(
                accept.sum()
            ),

        "accepted_prevalence":
            accepted[
                "prevalence"
            ],

        "accepted_accuracy":
            accepted[
                "accuracy"
            ],

        "accepted_balanced_accuracy":
            accepted[
                "balanced_accuracy"
            ],

        "accepted_precision":
            accepted[
                "precision"
            ],

        "accepted_recall":
            accepted[
                "recall"
            ],

        "accepted_f1":
            accepted[
                "f1"
            ],

        "accepted_roc_auc":
            accepted[
                "roc_auc"
            ],

        "accepted_pr_auc":
            accepted[
                "pr_auc"
            ],

        "accepted_brier":
            accepted[
                "brier"
            ],

        # selective risk
        "accepted_error_rate":
            accepted[
                "error_rate"
            ],

        "abstained_error_rate":
            abstained[
                "error_rate"
            ]
    })


gate_curve = pd.DataFrame(
    gate_rows
)

print("\n")
print("=" * 110)
print("SELECTIVE GATE SENSITIVITY — 2025 FORWARD")
print("=" * 110)

print(
    gate_curve
    .round(6)
    .to_string(
        index=False
    )
)


# ============================================================
# 21. Bootstrap uncertainty for PRIMARY gating policy
#
# Bootstrap records with replacement and reapply the gate.
# Primary quantities:
#   coverage
#   accepted error rate
#   abstained error rate
#   error-risk ratio
#   accepted F1
#   F1 change from all records
# ============================================================

def bootstrap_gate(
    y,
    prob,
    rri_scores,
    threshold,
    rri_cutoff,
    n_bootstrap=500,
    seed=42
):

    y = np.asarray(
        y
    )

    prob = np.asarray(
        prob
    )

    rri_scores = np.asarray(
        rri_scores
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

        yy = y[
            idx
        ]

        pp = prob[
            idx
        ]

        rr = rri_scores[
            idx
        ]

        pred = (
            pp >= threshold
        ).astype(int)

        accept = (
            rr > rri_cutoff
        )

        abstain = (
            ~accept
        )

        if (
            accept.sum() == 0
            or abstain.sum() == 0
        ):
            continue

        overall_f1 = f1_score(
            yy,
            pred,
            zero_division=0
        )

        accepted_f1 = f1_score(
            yy[
                accept
            ],
            pred[
                accept
            ],
            zero_division=0
        )

        accepted_error = np.mean(
            pred[
                accept
            ]
            !=
            yy[
                accept
            ]
        )

        abstained_error = np.mean(
            pred[
                abstain
            ]
            !=
            yy[
                abstain
            ]
        )

        rows.append({

            "coverage":
                accept.mean(),

            "abstention_rate":
                abstain.mean(),

            "overall_f1":
                overall_f1,

            "accepted_f1":
                accepted_f1,

            "delta_f1":
                (
                    accepted_f1 -
                    overall_f1
                ),

            "accepted_error_rate":
                accepted_error,

            "abstained_error_rate":
                abstained_error,

            "error_rate_difference":
                (
                    abstained_error -
                    accepted_error
                ),

            "error_risk_ratio":
                (
                    abstained_error /
                    accepted_error
                )
                if accepted_error > 0
                else np.nan
        })

        if (
            (b + 1) % 100 == 0
        ):
            print(
                f"Bootstrap "
                f"{b+1}/{n_bootstrap}"
            )

    return pd.DataFrame(
        rows
    )


print("\nBootstrapping primary gate...")

gate_boot = bootstrap_gate(
    y_forward,
    base_forward_prob,
    rri_scores_forward,
    base_threshold,
    PRIMARY_RRI_CUTOFF,
    n_bootstrap=500,
    seed=42
)


# ============================================================
# 22. Bootstrap CI summary
# ============================================================

def bootstrap_summary(
    df
):

    rows = []

    for col in [
        "coverage",
        "abstention_rate",
        "overall_f1",
        "accepted_f1",
        "delta_f1",
        "accepted_error_rate",
        "abstained_error_rate",
        "error_rate_difference",
        "error_risk_ratio"
    ]:

        values = (
            df[
                col
            ]
            .replace(
                [
                    np.inf,
                    -np.inf
                ],
                np.nan
            )
            .dropna()
        )

        rows.append({

            "metric":
                col,

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

    return pd.DataFrame(
        rows
    )


gate_ci = bootstrap_summary(
    gate_boot
)

print("\n")
print("=" * 110)
print("PRIMARY GATE BOOTSTRAP 95% CONFIDENCE INTERVALS")
print("=" * 110)

print(
    gate_ci
    .round(6)
    .to_string(
        index=False
    )
)


# ============================================================
# 23. Save outputs
# ============================================================

model_results.to_csv(
    OUT /
    "forward_downstream_model_results.csv",
    index=False
)

difference_df.to_csv(
    OUT /
    "forward_downstream_incremental_differences.csv",
    index=False
)

primary_gate_summary.to_csv(
    OUT /
    "primary_selective_gate_summary.csv",
    index=False
)

error_concentration.to_csv(
    OUT /
    "selective_gate_error_concentration.csv",
    index=False
)

gate_curve.to_csv(
    OUT /
    "selective_gate_sensitivity.csv",
    index=False
)

gate_boot.to_csv(
    OUT /
    "primary_gate_bootstrap_replicates.csv",
    index=False
)

gate_ci.to_csv(
    OUT /
    "primary_gate_bootstrap_confidence_intervals.csv",
    index=False
)

base_search.to_csv(
    OUT /
    "baseline_threshold_search.csv",
    index=False
)

rri_search.to_csv(
    OUT /
    "rri_threshold_search.csv",
    index=False
)


# ============================================================
# 24. Configuration
# ============================================================

configuration = {

    "analysis":
        "Forward downstream high-acuity validation and selective prediction",

    "development_year":
        2024,

    "forward_validation_year":
        2025,

    "primary_high_acuity_codes": {
        "4219001":
            "Critical",
        "4219003":
            "Emergent"
    },

    "negative_code": {
        "4219005":
            "Lower Acuity"
    },

    "dead_without_resuscitation":
        "Handled separately in Analysis 07 sensitivity analysis",

    "baseline_predictors":
        baseline_predictors,

    "baseline_predictor_count":
        18,

    "ems_rri_predictor_count":
        19,

    "primary_gate_cutoff":
        PRIMARY_RRI_CUTOFF,

    "gate_rule":
        "Accept if EMS-RRI > cutoff; abstain if EMS-RRI <= cutoff",

    "gate_sensitivity_cutoffs":
        GATE_CUTOFFS,

    "threshold_selection":
        "Maximum F1 on 2024 validation set",

    "bootstrap_replicates":
        500,

    "random_state":
        42
}

with open(
    OUT /
    "analysis_configuration.json",
    "w"
) as f:

    json.dump(
        configuration,
        f,
        indent=2
    )


print("\nSaved outputs to:")
print(OUT)

# Next original cell

# ============================================================
# 25. Paired bootstrap uncertainty for FORWARD downstream
#     baseline vs baseline + EMS-RRI
#
# Same records are sampled for both models.
# Positive delta = EMS-RRI model is higher,
# except Brier/error_rate where negative is better.
# ============================================================

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss
)


def paired_metric_values(
    y,
    prob,
    threshold
):

    pred = (
        prob >= threshold
    ).astype(int)

    return {
        "accuracy":
            accuracy_score(y, pred),

        "balanced_accuracy":
            balanced_accuracy_score(
                y,
                pred
            ),

        "precision":
            precision_score(
                y,
                pred,
                zero_division=0
            ),

        "recall":
            recall_score(
                y,
                pred,
                zero_division=0
            ),

        "f1":
            f1_score(
                y,
                pred,
                zero_division=0
            ),

        "roc_auc":
            roc_auc_score(
                y,
                prob
            ),

        "pr_auc":
            average_precision_score(
                y,
                prob
            ),

        "brier":
            brier_score_loss(
                y,
                prob
            ),

        "error_rate":
            np.mean(
                pred != y
            )
    }


def paired_bootstrap_models(
    y,
    base_prob,
    rri_prob,
    base_threshold,
    rri_threshold,
    n_bootstrap=500,
    seed=42
):

    y = np.asarray(y)
    base_prob = np.asarray(base_prob)
    rri_prob = np.asarray(rri_prob)

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

        yy = y[idx]

        if len(
            np.unique(yy)
        ) < 2:
            continue

        base = paired_metric_values(
            yy,
            base_prob[idx],
            base_threshold
        )

        rri = paired_metric_values(
            yy,
            rri_prob[idx],
            rri_threshold
        )

        row = {
            "bootstrap_iteration":
                b + 1
        }

        for metric in base.keys():

            row[
                f"delta_{metric}"
            ] = (
                rri[metric]
                -
                base[metric]
            )

        rows.append(
            row
        )

        if (
            (b + 1) % 100 == 0
        ):
            print(
                f"Bootstrap "
                f"{b+1}/{n_bootstrap}"
            )

    return pd.DataFrame(
        rows
    )


print("\nPaired bootstrap — 2024 internal test")

boot_model_internal = (
    paired_bootstrap_models(
        y_test,
        base_test_prob,
        rri_test_prob,
        base_threshold,
        rri_threshold,
        n_bootstrap=500,
        seed=42
    )
)


print("\nPaired bootstrap — 2025 forward")

boot_model_forward = (
    paired_bootstrap_models(
        y_forward,
        base_forward_prob,
        rri_forward_prob,
        base_threshold,
        rri_threshold,
        n_bootstrap=500,
        seed=43
    )
)

# Next original cell

# ============================================================
# 26. Summarize paired model bootstrap CIs
# ============================================================

PAIR_METRICS = [
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "brier",
    "error_rate"
]


def get_point_differences(
    y,
    base_prob,
    rri_prob,
    base_threshold,
    rri_threshold
):

    base = paired_metric_values(
        y,
        base_prob,
        base_threshold
    )

    rri = paired_metric_values(
        y,
        rri_prob,
        rri_threshold
    )

    return {
        metric:
            rri[metric]
            -
            base[metric]
        for metric in PAIR_METRICS
    }


def summarize_paired_bootstrap(
    dataset,
    boot_df,
    point_diff
):

    rows = []

    for metric in PAIR_METRICS:

        values = (
            boot_df[
                f"delta_{metric}"
            ]
            .dropna()
        )

        rows.append({

            "dataset":
                dataset,

            "metric":
                metric,

            "difference":
                point_diff[
                    metric
                ],

            "lower_95":
                values.quantile(
                    0.025
                ),

            "upper_95":
                values.quantile(
                    0.975
                ),

            "bootstrap_p":
                min(
                    1.0,
                    2 * min(
                        np.mean(
                            values <= 0
                        ),
                        np.mean(
                            values >= 0
                        )
                    )
                )
        })

    return pd.DataFrame(
        rows
    )


point_internal = get_point_differences(
    y_test,
    base_test_prob,
    rri_test_prob,
    base_threshold,
    rri_threshold
)

point_forward = get_point_differences(
    y_forward,
    base_forward_prob,
    rri_forward_prob,
    base_threshold,
    rri_threshold
)


paired_model_ci = pd.concat(
    [
        summarize_paired_bootstrap(
            "2024 internal test",
            boot_model_internal,
            point_internal
        ),

        summarize_paired_bootstrap(
            "2025 forward temporal",
            boot_model_forward,
            point_forward
        )
    ],
    ignore_index=True
)


print("\n")
print("=" * 110)
print("FORWARD DOWNSTREAM PAIRED BOOTSTRAP DIFFERENCES")
print("=" * 110)

print(
    paired_model_ci
    .round(6)
    .to_string(
        index=False
    )
)

# Next original cell

# ============================================================
# 27. Bootstrap uncertainty for ALL selective-gate cutoffs
#
# Important:
# These thresholds were pre-specified from the EMS-RRI
# cutoff-sensitivity analysis and are NOT optimized against
# downstream high-acuity errors.
# ============================================================

def bootstrap_gate_cutoff(
    y,
    prob,
    rri_scores,
    model_threshold,
    rri_cutoff,
    n_bootstrap=500,
    seed=42
):

    y = np.asarray(y)
    prob = np.asarray(prob)
    rri_scores = np.asarray(
        rri_scores
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

        yy = y[idx]
        pp = prob[idx]
        rr = rri_scores[idx]

        pred = (
            pp >= model_threshold
        ).astype(int)

        accept = (
            rr > rri_cutoff
        )

        abstain = (
            ~accept
        )

        if (
            accept.sum() == 0
            or abstain.sum() == 0
        ):
            continue

        overall_f1 = f1_score(
            yy,
            pred,
            zero_division=0
        )

        accepted_f1 = f1_score(
            yy[accept],
            pred[accept],
            zero_division=0
        )

        accepted_error = np.mean(
            pred[accept]
            !=
            yy[accept]
        )

        abstained_error = np.mean(
            pred[abstain]
            !=
            yy[abstain]
        )

        rows.append({

            "bootstrap_iteration":
                b + 1,

            "rri_cutoff":
                rri_cutoff,

            "coverage":
                accept.mean(),

            "abstention_rate":
                abstain.mean(),

            "accepted_f1":
                accepted_f1,

            "overall_f1":
                overall_f1,

            "delta_f1":
                accepted_f1
                -
                overall_f1,

            "accepted_error_rate":
                accepted_error,

            "abstained_error_rate":
                abstained_error,

            "error_rate_difference":
                abstained_error
                -
                accepted_error,

            "error_risk_ratio":
                (
                    abstained_error
                    /
                    accepted_error
                )
                if accepted_error > 0
                else np.nan
        })

    return pd.DataFrame(
        rows
    )


gate_boot_all = []

for i, cutoff in enumerate(
    GATE_CUTOFFS
):

    print(
        "\nBootstrapping gate cutoff:",
        cutoff
    )

    temp = bootstrap_gate_cutoff(
        y_forward,
        base_forward_prob,
        rri_scores_forward,
        base_threshold,
        cutoff,
        n_bootstrap=500,
        seed=100 + i
    )

    gate_boot_all.append(
        temp
    )


gate_boot_all = pd.concat(
    gate_boot_all,
    ignore_index=True
)

# Next original cell

# ============================================================
# 28. Gate sensitivity confidence intervals
# ============================================================

GATE_METRICS = [
    "coverage",
    "abstention_rate",
    "delta_f1",
    "accepted_error_rate",
    "abstained_error_rate",
    "error_rate_difference",
    "error_risk_ratio"
]


gate_ci_rows = []

for cutoff in GATE_CUTOFFS:

    subset = (
        gate_boot_all[
            gate_boot_all[
                "rri_cutoff"
            ] == cutoff
        ]
    )

    for metric in GATE_METRICS:

        values = (
            subset[
                metric
            ]
            .replace(
                [
                    np.inf,
                    -np.inf
                ],
                np.nan
            )
            .dropna()
        )

        gate_ci_rows.append({

            "rri_cutoff":
                cutoff,

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


gate_sensitivity_ci = pd.DataFrame(
    gate_ci_rows
)


print("\n")
print("=" * 110)
print("SELECTIVE-GATE SENSITIVITY BOOTSTRAP CIs")
print("=" * 110)

print(
    gate_sensitivity_ci
    .round(6)
    .to_string(
        index=False
    )
)


# ============================================================
# 29. Save final Analysis-08 uncertainty outputs
# ============================================================

paired_model_ci.to_csv(
    OUT /
    "forward_downstream_paired_bootstrap_differences.csv",
    index=False
)

boot_model_internal.to_csv(
    OUT /
    "forward_model_bootstrap_2024_internal.csv",
    index=False
)

boot_model_forward.to_csv(
    OUT /
    "forward_model_bootstrap_2025.csv",
    index=False
)

gate_boot_all.to_csv(
    OUT /
    "all_gate_bootstrap_replicates.csv",
    index=False
)

gate_sensitivity_ci.to_csv(
    OUT /
    "selective_gate_sensitivity_bootstrap_cis.csv",
    index=False
)

print(
    "\nFinal Analysis 08 uncertainty outputs saved."
)