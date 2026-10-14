# 09 feature set selection
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
# EMS-RRI REVISION ANALYSIS 11
# 2024-Only Feature-Set Selection Check
#
# PURPOSE
# ------------------------------------------------------------
# Re-evaluate the original 34-predictor XGBoost specification
# against the final 32-predictor specification using ONLY the
# 2024 development cohort.
#
# This closes a temporal-design consistency issue:
#   - 2024 train: model fitting
#   - 2024 validation: feature-set comparison + threshold selection
#   - 2024 internal test: confirmatory evaluation
#   - 2025 full cohort: forward temporal evaluation
#
# IMPORTANT
# ------------------------------------------------------------
# 2025 is NOT used to choose between 34 and 32 predictors.
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
    "11_2024_feature_set_selection"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

print("Output directory:")
print(OUT)


# ============================================================
# 3. Load data
# ============================================================

df24 = pd.read_parquet(
    DATA_2024
)

df25 = pd.read_parquet(
    DATA_2025
)

print("\nDataset shapes:")
print("2024:", df24.shape)
print("2025:", df25.shape)


# ============================================================
# 4. Exact predictor specifications
# ============================================================

# -------------------------
# Numeric predictors
# -------------------------

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


# -------------------------
# Final 32-predictor
# categorical specification
# -------------------------

categorical_32 = [
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


# -------------------------
# Two predictors removed
# from the original 34 set
# -------------------------

removed_high_cardinality = [
    "primary_impression_eSituation_11",
    "primary_symptom_eSituation_09",
]


categorical_34 = (
    categorical_32
    +
    removed_high_cardinality
)


predictors_32 = (
    numeric_features
    +
    categorical_32
)

predictors_34 = (
    numeric_features
    +
    categorical_34
)


assert len(
    predictors_32
) == 32

assert len(
    predictors_34
) == 34


print("\nPredictor specifications:")
print(
    "32-predictor model:",
    len(predictors_32)
)

print(
    "34-predictor model:",
    len(predictors_34)
)

print(
    "Additional 34-model predictors:",
    removed_high_cardinality
)


# ============================================================
# 5. Verify all required columns exist
# ============================================================

required_columns = sorted(
    set(
        predictors_34
        +
        ["ems_rri_basic"]
    )
)

for year, df in [
    (2024, df24),
    (2025, df25)
]:

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:

        raise RuntimeError(
            f"{year}: required columns missing:\n"
            +
            "\n".join(
                missing
            )
        )

print(
    "\nAll required predictors verified."
)


# ============================================================
# 6. Recreate corrected 2024 development split
#
# Must match Analysis 01:
#
#   Train:         140,000
#   Validation:     30,000
#   Internal test:  30,000
#
# random_state = 42
# no stratification
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
print(
    f"Train:         {len(train24):,}"
)

print(
    f"Validation:    {len(val24):,}"
)

print(
    f"Internal test: {len(test24):,}"
)

print(
    f"2025 forward:  {len(df25):,}"
)


assert len(
    train24
) == 140000

assert len(
    val24
) == 30000

assert len(
    test24
) == 30000


# ============================================================
# 7. Derive low-reliability cutoff from 2024 TRAIN ONLY
# ============================================================

RRI_CUTOFF = float(
    train24[
        "ems_rri_basic"
    ].quantile(
        0.20
    )
)

print(
    "\n2024-training EMS-RRI cutoff:",
    RRI_CUTOFF
)


def make_target(
    df
):

    score = pd.to_numeric(
        df[
            "ems_rri_basic"
        ],
        errors="coerce"
    )

    return (
        score <= RRI_CUTOFF
    ).astype(int)


y_train = make_target(
    train24
).to_numpy()

y_val = make_target(
    val24
).to_numpy()

y_test = make_target(
    test24
).to_numpy()

y_forward = make_target(
    df25
).to_numpy()


print("\nLow-reliability prevalence:")

print(
    "2024 train:",
    round(
        y_train.mean(),
        6
    )
)

print(
    "2024 validation:",
    round(
        y_val.mean(),
        6
    )
)

print(
    "2024 internal:",
    round(
        y_test.mean(),
        6
    )
)

print(
    "2025 forward:",
    round(
        y_forward.mean(),
        6
    )
)


# ============================================================
# 8. Input preparation
# ============================================================

def prepare_X(
    df,
    predictors
):

    X = df[
        predictors
    ].copy()

    for column in numeric_features:

        if column in X.columns:

            X[column] = pd.to_numeric(
                X[column],
                errors="coerce"
            )

    return X


# ============================================================
# 9. Pipeline builder
# ============================================================

def build_pipeline(
    predictors
):

    numeric = [
        col
        for col in predictors
        if col in numeric_features
    ]

    categorical = [
        col
        for col in predictors
        if col not in numeric_features
    ]


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
                    handle_unknown="ignore",
                    sparse_output=True
                )
            )
        ]
    )


    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical
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
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                classifier
            )
        ]
    )


# ============================================================
# 10. Validation threshold selection
#
# Threshold chosen independently for each specification
# by maximum F1 on the 2024 validation partition.
# ============================================================

def select_f1_threshold(
    y_true,
    probability
):

    rows = []

    for threshold in np.arange(
        0.01,
        1.00,
        0.01
    ):

        prediction = (
            probability >=
            threshold
        ).astype(int)

        rows.append({
            "threshold":
                round(
                    float(
                        threshold
                    ),
                    2
                ),

            "precision":
                precision_score(
                    y_true,
                    prediction,
                    zero_division=0
                ),

            "recall":
                recall_score(
                    y_true,
                    prediction,
                    zero_division=0
                ),

            "f1":
                f1_score(
                    y_true,
                    prediction,
                    zero_division=0
                )
        })


    table = pd.DataFrame(
        rows
    )

    best_index = (
        table[
            "f1"
        ].idxmax()
    )

    best_threshold = float(
        table.loc[
            best_index,
            "threshold"
        ]
    )

    return (
        best_threshold,
        table
    )


# ============================================================
# 11. Evaluation helper
# ============================================================

def evaluate(
    y_true,
    probability,
    threshold
):

    prediction = (
        probability >=
        threshold
    ).astype(int)

    tn, fp, fn, tp = (
        confusion_matrix(
            y_true,
            prediction,
            labels=[0, 1]
        ).ravel()
    )


    return {
        "accuracy":
            accuracy_score(
                y_true,
                prediction
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                y_true,
                prediction
            ),

        "precision":
            precision_score(
                y_true,
                prediction,
                zero_division=0
            ),

        "recall":
            recall_score(
                y_true,
                prediction,
                zero_division=0
            ),

        "f1":
            f1_score(
                y_true,
                prediction,
                zero_division=0
            ),

        "roc_auc":
            roc_auc_score(
                y_true,
                probability
            ),

        "pr_auc":
            average_precision_score(
                y_true,
                probability
            ),

        "brier":
            brier_score_loss(
                y_true,
                probability
            ),

        "fpr":
            fp /
            (
                fp + tn
            ),

        "fnr":
            fn /
            (
                fn + tp
            )
    }


# ============================================================
# 12. Run both specifications
# ============================================================

MODEL_SPECS = {
    "XGBoost-34":
        predictors_34,

    "XGBoost-32":
        predictors_32
}


results = []

threshold_tables = []

models = {}

probability_store = {}


for model_name, predictors in (
    MODEL_SPECS.items()
):

    print(
        "\n"
        +
        "=" * 90
    )

    print(
        model_name
    )

    print(
        "=" * 90
    )


    X_train = prepare_X(
        train24,
        predictors
    )

    X_val = prepare_X(
        val24,
        predictors
    )

    X_test = prepare_X(
        test24,
        predictors
    )

    X_forward = prepare_X(
        df25,
        predictors
    )


    model = build_pipeline(
        predictors
    )


    print(
        f"Training {model_name}..."
    )

    model.fit(
        X_train,
        y_train
    )

    print(
        "Training complete."
    )


    val_probability = (
        model
        .predict_proba(
            X_val
        )[:, 1]
    )


    threshold, threshold_table = (
        select_f1_threshold(
            y_val,
            val_probability
        )
    )


    threshold_table[
        "model"
    ] = model_name

    threshold_tables.append(
        threshold_table
    )


    print(
        "Selected 2024-validation threshold:",
        threshold
    )


    test_probability = (
        model
        .predict_proba(
            X_test
        )[:, 1]
    )

    forward_probability = (
        model
        .predict_proba(
            X_forward
        )[:, 1]
    )


    probability_store[
        model_name
    ] = {
        "validation":
            val_probability,

        "internal_test":
            test_probability,

        "forward":
            forward_probability
    }


    for (
        dataset_name,
        y_true,
        probability
    ) in [

        (
            "2024 validation",
            y_val,
            val_probability
        ),

        (
            "2024 internal test",
            y_test,
            test_probability
        ),

        (
            "2025 forward temporal",
            y_forward,
            forward_probability
        )
    ]:

        metrics = evaluate(
            y_true,
            probability,
            threshold
        )


        results.append({
            "model":
                model_name,

            "predictor_count":
                len(
                    predictors
                ),

            "threshold":
                threshold,

            "dataset":
                dataset_name,

            **metrics
        })


    models[
        model_name
    ] = model


# ============================================================
# 13. Results table
# ============================================================

results_df = pd.DataFrame(
    results
)

threshold_search_df = pd.concat(
    threshold_tables,
    ignore_index=True
)


print("\n")
print("=" * 120)
print("34 vs 32 PREDICTOR COMPARISON")
print("=" * 120)

print(
    results_df[
        [
            "model",
            "predictor_count",
            "dataset",
            "threshold",
            "roc_auc",
            "pr_auc",
            "f1",
            "precision",
            "recall",
            "brier"
        ]
    ]
    .round(6)
    .to_string(
        index=False
    )
)


# ============================================================
# 14. Direct 32-minus-34 differences
#
# Positive:
#   32-model performs better
#
# For Brier:
#   negative is better
# ============================================================

comparison_rows = []


for dataset_name in [
    "2024 validation",
    "2024 internal test",
    "2025 forward temporal"
]:

    subset = (
        results_df[
            results_df[
                "dataset"
            ] == dataset_name
        ]
        .set_index(
            "model"
        )
    )


    comparison_rows.append({
        "dataset":
            dataset_name,

        "delta_roc_auc_32_minus_34":
            (
                subset.loc[
                    "XGBoost-32",
                    "roc_auc"
                ]
                -
                subset.loc[
                    "XGBoost-34",
                    "roc_auc"
                ]
            ),

        "delta_pr_auc_32_minus_34":
            (
                subset.loc[
                    "XGBoost-32",
                    "pr_auc"
                ]
                -
                subset.loc[
                    "XGBoost-34",
                    "pr_auc"
                ]
            ),

        "delta_f1_32_minus_34":
            (
                subset.loc[
                    "XGBoost-32",
                    "f1"
                ]
                -
                subset.loc[
                    "XGBoost-34",
                    "f1"
                ]
            ),

        "delta_brier_32_minus_34":
            (
                subset.loc[
                    "XGBoost-32",
                    "brier"
                ]
                -
                subset.loc[
                    "XGBoost-34",
                    "brier"
                ]
            )
    })


comparison_df = pd.DataFrame(
    comparison_rows
)


print("\n")
print("=" * 120)
print("32-PREDICTOR MINUS 34-PREDICTOR PERFORMANCE")
print("Positive ROC/PR/F1 = 32-predictor model is better")
print("Negative Brier = 32-predictor model is better")
print("=" * 120)

print(
    comparison_df
    .round(6)
    .to_string(
        index=False
    )
)


# ============================================================
# 15. Selection decision based ONLY on 2024 validation
# ============================================================

validation_results = (
    results_df[
        results_df[
            "dataset"
        ] == "2024 validation"
    ]
    .set_index(
        "model"
    )
)


selection_summary = {
    "selection_dataset":
        "2024 validation",

    "xgb34_roc_auc":
        float(
            validation_results.loc[
                "XGBoost-34",
                "roc_auc"
            ]
        ),

    "xgb32_roc_auc":
        float(
            validation_results.loc[
                "XGBoost-32",
                "roc_auc"
            ]
        ),

    "xgb34_pr_auc":
        float(
            validation_results.loc[
                "XGBoost-34",
                "pr_auc"
            ]
        ),

    "xgb32_pr_auc":
        float(
            validation_results.loc[
                "XGBoost-32",
                "pr_auc"
            ]
        ),

    "xgb34_f1":
        float(
            validation_results.loc[
                "XGBoost-34",
                "f1"
            ]
        ),

    "xgb32_f1":
        float(
            validation_results.loc[
                "XGBoost-32",
                "f1"
            ]
        ),

    "xgb34_threshold":
        float(
            validation_results.loc[
                "XGBoost-34",
                "threshold"
            ]
        ),

    "xgb32_threshold":
        float(
            validation_results.loc[
                "XGBoost-32",
                "threshold"
            ]
        )
}


print("\n")
print("=" * 120)
print("2024-ONLY FEATURE-SELECTION SUMMARY")
print("=" * 120)

for key, value in (
    selection_summary.items()
):

    print(
        f"{key}: {value}"
    )


# ============================================================
# 16. Save
# ============================================================

results_df.to_csv(
    OUT /
    "xgboost_34_vs_32_all_metrics.csv",
    index=False
)

comparison_df.to_csv(
    OUT /
    "xgboost_32_minus_34_differences.csv",
    index=False
)

threshold_search_df.to_csv(
    OUT /
    "validation_threshold_searches.csv",
    index=False
)


with open(
    OUT /
    "feature_selection_summary.json",
    "w"
) as f:

    json.dump(
        selection_summary,
        f,
        indent=2
    )


configuration = {
    "purpose":
        "2024-only feature-set selection consistency check",

    "development_year":
        2024,

    "forward_validation_year":
        2025,

    "train_n":
        len(train24),

    "validation_n":
        len(val24),

    "internal_test_n":
        len(test24),

    "forward_n":
        len(df25),

    "ems_rri_cutoff":
        RRI_CUTOFF,

    "random_state":
        42,

    "removed_predictors":
        removed_high_cardinality,

    "selection_rule":
        (
            "Compare XGBoost-34 and XGBoost-32 on "
            "2024 validation only; 2025 forward results "
            "are not used for feature-set selection."
        )
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