# 05 domain ablation
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
# EMS-RRI REVISION ANALYSIS 06
# Predictor-Domain Group Ablation
#
# Reviewer concern:
# Aggregated SHAP importance may favor domains containing
# more predictors. Evaluate domain importance using
# leave-one-domain-out feature-group ablation.
#
# Design:
#   Development: 2024
#       70% train
#       15% validation
#       15% internal test
#
#   Forward temporal validation:
#       Full 2025 cohort
#
# For full model + each domain ablation:
#   1. Fit preprocessing/model on 2024 training only
#   2. Select F1-optimal threshold on 2024 validation
#   3. Evaluate 2024 internal test
#   4. Evaluate 2025 forward temporal cohort
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

OUT = (
    f"{BASE}/revision_analysis/outputs/"
    "06_shap_group_analysis"
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

numeric_features_full = [
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

categorical_features_full = [
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

all_predictors = (
    numeric_features_full +
    categorical_features_full
)

assert len(all_predictors) == 32

print("\nFull predictor count:", len(all_predictors))


# ============================================================
# 5. Define predictor domains
#
# Counts should sum to 32.
# ============================================================

domain_map = {

    "Timing": [
        "computed_EMSChuteTimeMin",
        "computed_EMSDispatchCenterTimeSec",
        "computed_EMSSceneResponseTimeMin",
        "computed_EMSSceneTimeMin",
        "computed_EMSSceneToPatientTimeMin",
        "computed_EMSSystemResponseTimeMin",
        "computed_EMSTotalCallTimeMin",
        "computed_EMSTransportTimeMin",
    ],

    "Geography": [
        "computed_NasemsoRegion",
        "computed_USCensusDivision",
        "computed_USCensusRegion",
        "computed_Urbanicity",
    ],

    "Patient": [
        "computed_ageinyear",
    ],

    "Clinical": [
        "eArrest_05",
        "eSituation_02",
        "eSituation_07",
        "eSituation_08",
        "eSituation_13",
        "eSituation_20",
    ],

    "Disposition": [
        "eDisposition_16",
        "eDisposition_17",
        "eDisposition_19",
        "eDisposition_21",
        "eDisposition_22",
        "eDisposition_28",
        "eDisposition_29",
        "eDisposition_30",
        "eDisposition_32",
    ],

    "Outcome": [
        "eOutcome_01",
        "eOutcome_02",
    ],

    "Payment": [
        "ePayment_01",
        "ePayment_50",
    ],
}


# ------------------------------------------------------------
# Verify domain mapping
# ------------------------------------------------------------

mapped_predictors = []

for domain, cols in domain_map.items():

    mapped_predictors.extend(cols)

    print(
        f"{domain:12s}: "
        f"{len(cols)} predictors"
    )

assert len(mapped_predictors) == 32
assert set(mapped_predictors) == set(all_predictors)

print("\nDomain mapping verified.")


# ============================================================
# 6. Recreate 2024 development split
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
# 7. Derive EMS-RRI target from 2024 TRAIN ONLY
# ============================================================

rri_cutoff = float(
    train24[
        "ems_rri_basic"
    ].quantile(0.20)
)

print("\nEMS-RRI cutoff:")
print(rri_cutoff)


def make_target(df, cutoff):

    score = pd.to_numeric(
        df["ems_rri_basic"],
        errors="coerce"
    )

    return (
        score <= cutoff
    ).astype(int)


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


# ============================================================
# 8. Build model for arbitrary feature set
# ============================================================

def build_pipeline(predictors):

    numeric_features = [
        c
        for c in predictors
        if c in numeric_features_full
    ]

    categorical_features = [
        c
        for c in predictors
        if c in categorical_features_full
    ]


    # --------------------------------------------------------
    # Numeric pipeline
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Categorical pipeline
    # --------------------------------------------------------

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


    transformers = []

    if len(numeric_features) > 0:

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_features
            )
        )

    if len(categorical_features) > 0:

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        )


    preprocessor = ColumnTransformer(
        transformers=transformers
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


    return (
        model,
        numeric_features,
        categorical_features
    )


# ============================================================
# 9. Prepare X for arbitrary feature set
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


# ============================================================
# 10. F1-optimal threshold selection
# ============================================================

def find_best_threshold(
    y_true,
    probabilities
):

    rows = []

    for threshold in np.arange(
        0.01,
        1.00,
        0.01
    ):

        pred = (
            probabilities >= threshold
        ).astype(int)

        rows.append({

            "threshold":
                round(
                    float(threshold),
                    2
                ),

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

    table = pd.DataFrame(
        rows
    )

    best_idx = table[
        "f1"
    ].idxmax()

    return (
        float(
            table.loc[
                best_idx,
                "threshold"
            ]
        ),
        table
    )


# ============================================================
# 11. Evaluation function
# ============================================================

def evaluate(
    y_true,
    probabilities,
    threshold
):

    pred = (
        probabilities >= threshold
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
                probabilities
            ),

        "pr_auc":
            average_precision_score(
                y_true,
                probabilities
            ),

        "brier":
            brier_score_loss(
                y_true,
                probabilities
            ),

        "fpr":
            fp / (fp + tn),

        "fnr":
            fn / (fn + tp),

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
# 12. Run full model + leave-one-domain-out models
# ============================================================

experiments = {
    "Full model":
        all_predictors
}

for domain, domain_features in domain_map.items():

    reduced = [
        feature
        for feature in all_predictors
        if feature not in domain_features
    ]

    experiments[
        f"Without {domain}"
    ] = reduced


results = []
threshold_searches = []


for i, (experiment_name, predictors) in enumerate(
    experiments.items(),
    start=1
):

    print("\n" + "=" * 85)
    print(
        f"RUN {i}/{len(experiments)}: "
        f"{experiment_name}"
    )
    print("=" * 85)

    removed_domain = (
        "None"
        if experiment_name == "Full model"
        else experiment_name.replace(
            "Without ",
            ""
        )
    )

    removed_count = (
        0
        if removed_domain == "None"
        else len(
            domain_map[
                removed_domain
            ]
        )
    )


    # --------------------------------------------------------
    # Build pipeline
    # --------------------------------------------------------

    model, numeric_features, categorical_features = (
        build_pipeline(
            predictors
        )
    )


    # --------------------------------------------------------
    # Prepare matrices
    # --------------------------------------------------------

    X_train = prepare_X(
        train24,
        predictors,
        numeric_features
    )

    X_val = prepare_X(
        val24,
        predictors,
        numeric_features
    )

    X_test = prepare_X(
        test24,
        predictors,
        numeric_features
    )

    X_2025 = prepare_X(
        df25,
        predictors,
        numeric_features
    )


    print(
        "Predictors remaining:",
        len(predictors)
    )

    print(
        "Removed predictors:",
        removed_count
    )


    # --------------------------------------------------------
    # Fit model
    # --------------------------------------------------------

    print("Training...")

    model.fit(
        X_train,
        y_train
    )

    print("Training complete.")


    # --------------------------------------------------------
    # Validation threshold
    # --------------------------------------------------------

    val_prob = model.predict_proba(
        X_val
    )[:, 1]

    best_threshold, threshold_table = (
        find_best_threshold(
            y_val,
            val_prob
        )
    )

    print(
        "Selected threshold:",
        best_threshold
    )


    threshold_table[
        "experiment"
    ] = experiment_name

    threshold_searches.append(
        threshold_table
    )


    # --------------------------------------------------------
    # Predictions
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

    internal = evaluate(
        y_test,
        test_prob,
        best_threshold
    )

    internal.update({

        "experiment":
            experiment_name,

        "removed_domain":
            removed_domain,

        "removed_predictor_count":
            removed_count,

        "remaining_predictor_count":
            len(predictors),

        "selected_threshold":
            best_threshold,

        "dataset":
            "2024 internal test"
    })

    results.append(
        internal
    )


    # --------------------------------------------------------
    # Evaluate forward 2025
    # --------------------------------------------------------

    forward = evaluate(
        y_2025,
        forward_prob,
        best_threshold
    )

    forward.update({

        "experiment":
            experiment_name,

        "removed_domain":
            removed_domain,

        "removed_predictor_count":
            removed_count,

        "remaining_predictor_count":
            len(predictors),

        "selected_threshold":
            best_threshold,

        "dataset":
            "2025 forward temporal"
    })

    results.append(
        forward
    )


    print(
        "\n2024 internal: "
        f"ROC-AUC={internal['roc_auc']:.4f}, "
        f"PR-AUC={internal['pr_auc']:.4f}, "
        f"F1={internal['f1']:.4f}"
    )

    print(
        "2025 forward:  "
        f"ROC-AUC={forward['roc_auc']:.4f}, "
        f"PR-AUC={forward['pr_auc']:.4f}, "
        f"F1={forward['f1']:.4f}"
    )


# ============================================================
# 13. Combine results
# ============================================================

results_df = pd.DataFrame(
    results
)

threshold_search_df = pd.concat(
    threshold_searches,
    ignore_index=True
)

pd.set_option(
    "display.max_columns",
    None
)

pd.set_option(
    "display.width",
    220
)


# ============================================================
# 14. Extract full-model reference metrics
# ============================================================

full_reference = (
    results_df[
        results_df[
            "experiment"
        ] == "Full model"
    ]
    .set_index(
        "dataset"
    )
)


# ============================================================
# 15. Calculate performance loss after domain removal
#
# Positive loss = removing the domain made performance worse.
# ============================================================

ablation_rows = []

for _, row in results_df.iterrows():

    if row[
        "experiment"
    ] == "Full model":
        continue

    dataset = row[
        "dataset"
    ]

    full = full_reference.loc[
        dataset
    ]

    ablation_rows.append({

        "dataset":
            dataset,

        "removed_domain":
            row[
                "removed_domain"
            ],

        "removed_predictor_count":
            row[
                "removed_predictor_count"
            ],

        "remaining_predictor_count":
            row[
                "remaining_predictor_count"
            ],

        "model_threshold":
            row[
                "selected_threshold"
            ],

        "roc_auc":
            row[
                "roc_auc"
            ],

        "roc_auc_loss":
            full[
                "roc_auc"
            ]
            -
            row[
                "roc_auc"
            ],

        "pr_auc":
            row[
                "pr_auc"
            ],

        "pr_auc_loss":
            full[
                "pr_auc"
            ]
            -
            row[
                "pr_auc"
            ],

        "f1":
            row[
                "f1"
            ],

        "f1_loss":
            full[
                "f1"
            ]
            -
            row[
                "f1"
            ],

        "recall":
            row[
                "recall"
            ],

        "recall_loss":
            full[
                "recall"
            ]
            -
            row[
                "recall"
            ],

        "brier":
            row[
                "brier"
            ],

        # Positive = calibration/probability error worsened
        "brier_increase":
            row[
                "brier"
            ]
            -
            full[
                "brier"
            ]
    })


ablation_df = pd.DataFrame(
    ablation_rows
)


# ============================================================
# 16. Forward-validation group-ablation ranking
# ============================================================

forward_ablation = (
    ablation_df[
        ablation_df[
            "dataset"
        ] == "2025 forward temporal"
    ]
    .sort_values(
        "roc_auc_loss",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)

print("\n")
print("=" * 100)
print("2025 FORWARD DOMAIN ABLATION")
print("Positive loss means performance decreased after domain removal.")
print("=" * 100)

print(
    forward_ablation[
        [
            "removed_domain",
            "removed_predictor_count",
            "roc_auc",
            "roc_auc_loss",
            "pr_auc",
            "pr_auc_loss",
            "f1",
            "f1_loss",
            "recall",
            "recall_loss",
            "brier_increase"
        ]
    ]
    .round(6)
    .to_string(
        index=False
    )
)


# ============================================================
# 17. Full-model reference
# ============================================================

print("\n")
print("=" * 100)
print("FULL MODEL REFERENCE")
print("=" * 100)

print(
    results_df[
        results_df[
            "experiment"
        ] == "Full model"
    ][
        [
            "dataset",
            "selected_threshold",
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
# 18. Normalize ablation loss by number of predictors removed
#
# Secondary descriptive measure only.
# This helps distinguish a large group effect caused partly
# by simply containing many predictors.
# ============================================================

forward_ablation[
    "roc_auc_loss_per_predictor"
] = (
    forward_ablation[
        "roc_auc_loss"
    ]
    /
    forward_ablation[
        "removed_predictor_count"
    ]
)

forward_ablation[
    "pr_auc_loss_per_predictor"
] = (
    forward_ablation[
        "pr_auc_loss"
    ]
    /
    forward_ablation[
        "removed_predictor_count"
    ]
)

print("\n")
print("=" * 100)
print("FORWARD ABLATION NORMALIZED BY DOMAIN SIZE")
print("=" * 100)

print(
    forward_ablation[
        [
            "removed_domain",
            "removed_predictor_count",
            "roc_auc_loss",
            "roc_auc_loss_per_predictor",
            "pr_auc_loss",
            "pr_auc_loss_per_predictor"
        ]
    ]
    .sort_values(
        "roc_auc_loss_per_predictor",
        ascending=False
    )
    .round(6)
    .to_string(
        index=False
    )
)


# ============================================================
# 19. Save outputs
# ============================================================

results_df.to_csv(
    f"{OUT}/all_group_ablation_metrics.csv",
    index=False
)

ablation_df.to_csv(
    f"{OUT}/group_ablation_performance_losses.csv",
    index=False
)

forward_ablation.to_csv(
    f"{OUT}/forward_group_ablation_ranking.csv",
    index=False
)

threshold_search_df.to_csv(
    f"{OUT}/threshold_searches.csv",
    index=False
)


# ============================================================
# 20. Save domain map and configuration
# ============================================================

configuration = {

    "analysis":
        "Leave-one-domain-out feature-group ablation",

    "development_year":
        2024,

    "forward_validation_year":
        2025,

    "ems_rri_cutoff":
        rri_cutoff,

    "predictor_count":
        len(all_predictors),

    "domain_map":
        domain_map,

    "threshold_selection":
        "Maximum F1 on 2024 validation set for each model",

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
    f"{OUT}/group_ablation_configuration.json",
    "w"
) as f:

    json.dump(
        configuration,
        f,
        indent=2
    )


print("\nSaved outputs to:")
print(OUT)